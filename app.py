from visualization import plot_dg, visualize_sampler_run, supervise_3d_complex_app


app = supervise_3d_complex_app(
    pdb_name='6fff',
    folder_complexes='/home/giacomo/Documents/LCP_runs/RUN_1/docking/complexes',
    json_file='/home/giacomo/Documents/LCP_runs/RUN_1/docking_res.json',
    color_col='step',
    smiles_col='smiles',
    title_col='id'
)

app.run_server(debug=True)
