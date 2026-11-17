import os
import sys
import json
import io
import ast
import base64
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def run_notebook(nb_path):
    abs_nb_path = os.path.abspath(nb_path)
    print(f"=== Executing {abs_nb_path} ===")
    with open(abs_nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Change working directory to the notebook's directory so relative paths work
    nb_dir = os.path.dirname(abs_nb_path)
    orig_cwd = os.getcwd()
    os.chdir(nb_dir)

    namespace = {}
    exec_count = 1

    try:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            
            src = ''.join(cell.get('source', []))
            if not src.strip():
                continue

            print(f"Running cell {i}...")
            cell['execution_count'] = exec_count
            cell['outputs'] = []

            # Capture stdout
            stdout_io = io.StringIO()
            
            # Parse code into body and optional last expression
            tree = ast.parse(src)
            last_expr = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last_expr = ast.Expression(tree.body.pop().value)

            plt.close('all')

            with contextlib.redirect_stdout(stdout_io):
                if tree.body:
                    exec(compile(tree, f'<cell_{i}>', 'exec'), namespace)
                
                eval_res = None
                if last_expr is not None:
                    eval_res = eval(compile(last_expr, f'<cell_{i}>', 'eval'), namespace)

            stdout_str = stdout_io.getvalue()
            if stdout_str:
                cell['outputs'].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [line + '\n' for line in stdout_str.splitlines()]
                })

            # Check if any matplotlib figures were created
            fig_nums = plt.get_fignums()
            if fig_nums:
                for fignum in fig_nums:
                    fig = plt.figure(fignum)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
                    cell['outputs'].append({
                        "data": {
                            "image/png": img_b64,
                            "text/plain": ["<Figure size ...>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })

            # Format eval result if any
            if eval_res is not None:
                if isinstance(eval_res, pd.DataFrame):
                    cell['outputs'].append({
                        "data": {
                            "text/html": [eval_res.to_html()],
                            "text/plain": [str(eval_res)]
                        },
                        "execution_count": exec_count,
                        "metadata": {},
                        "output_type": "execute_result"
                    })
                elif not isinstance(eval_res, type(None)):
                    cell['outputs'].append({
                        "data": {
                            "text/plain": [repr(eval_res)]
                        },
                        "execution_count": exec_count,
                        "metadata": {},
                        "output_type": "execute_result"
                    })

            exec_count += 1
            print(f"Cell {i} finished ({len(cell['outputs'])} outputs).")

        with open(abs_nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully saved {abs_nb_path} with outputs!")

    finally:
        os.chdir(orig_cwd)

if __name__ == '__main__':
    for p in sys.argv[1:]:
        run_notebook(p)
