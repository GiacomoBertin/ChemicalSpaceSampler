import base64
import copy
import textwrap
from io import BytesIO
from typing import Callable, Union, List, Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import py3Dmol
from dash import Input, Output, dcc, html, no_update
from jupyter_dash import JupyterDash
from pandas.core.groupby import DataFrameGroupBy
from plotly.graph_objects import Figure
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.decomposition import PCA
from sklearn import manifold
import redis
from Utility import LigandInfo
import plotly.graph_objects as go
import os.path as osp
import dash_bio as dashbio
import dash_bio.utils.ngl_parser as ngl_parser
from dash_bio.utils import PdbParser
from dash_bio.utils.mol3dviewer_styles_creator import ATOM_COLORS, CHAIN_COLORS, RESIDUE_COLORS, RESIDUE_TYPE_COLORS, AMINO_ACID_CLASSES
from prody import parsePDBHeader
import json
import plotly
import dash
from dash import Dash, dcc, html


def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")


def transorm_MDS(molecules, n_dimensions, n_threads=-1):
    """
    Perform a Multidimensional scaling over a set of molecules. First compute the distance matrix, then perform
    MDS. It will return the coordinates in the new base
    :param n_threads:     Number or cores
    :param molecules:     An iterable with the input molecules
    :param n_dimensions:  Number of dimensions in which to immerse.
    :return:              The embedded coordinates
    """
    seed = np.random.RandomState(seed=3)
    nmds = manifold.MDS(n_components=n_dimensions, metric=False, max_iter=30000, eps=1e-5,
                        dissimilarity="precomputed", n_jobs=n_threads,
                        n_init=1)

    # first generate the distance matrix:
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 1024) for x in molecules]
    dists = []
    print("Generate the distance matrix\n")

    for i in range(0, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        dists.append([1 - x for x in sims])

    X = nmds.fit_transform(dists)
    return X


def test_groups(fig: Figure, df_grouped: DataFrameGroupBy):
    """Test if plotly figure curve names match up with pandas dataframe groups

    Args:
        fig (plotly figure): _description_
        groups (pandas groupby object): _description_

    Returns:
        _type_: Bool describing whether or not groups is the correct dataframe grouping descbrining the data in fig
    """
    str_groups = {}
    for name, group in df_grouped:
        if isinstance(name, tuple):
            str_groups[", ".join(str(x) for x in name)] = group
        else:
            str_groups[name] = group

    for data in fig.data:
        if data.name in str_groups:
            if len(data.y) == len(str_groups[data.name]):
                continue
        else:
            return False
    return True


def find_grouping(
        fig: Figure, df_data: pd.DataFrame, cols: List[str]
) -> Tuple[DataFrameGroupBy, dict]:
    if len(cols) == 1:
        df_grouped = df_data.groupby(cols)
        if not test_groups(fig, df_grouped):
            raise ValueError(
                "marker_col is misspecified because the dataframe grouping names don't match the names in the plotly figure."
            )

    elif len(cols) == 2:  # color_col and marker_col

        df_grouped_x = df_data.groupby(cols)
        df_grouped_y = df_data.groupby([cols[1], cols[0]])

        if test_groups(fig, df_grouped_x):
            df_grouped = df_grouped_x

        elif test_groups(fig, df_grouped_y):
            df_grouped = df_grouped_y
        else:
            raise ValueError(
                "color_col and marker_col are misspecified because their dataframe grouping names don't match the names in the plotly figure."
            )
    else:
        raise ValueError("Too many columns specified for grouping.")

    str_groups = {}
    for name, group in df_grouped:
        if isinstance(name, tuple):
            str_groups[", ".join(str(x) for x in name)] = group
        else:
            str_groups[name] = group

    curve_dict = {index: str_groups[x["name"]] for index, x in enumerate(fig.data)}
    return df_grouped, curve_dict


def add_molecules(
        fig: Figure,
        df: pd.DataFrame,
        smiles_col: Union[str, List[str]] = "SMILES",
        show_img: bool = True,
        svg_size: int = 200,
        alpha: float = 0.75,
        mol_alpha: float = 0.7,
        title_col: str = None,
        show_coords: bool = True,
        caption_cols: List[str] = None,
        caption_transform: Dict[str, Callable] = {},
        color_col: str = None,
        marker_col: str = None,
        wrap: bool = True,
        wraplen: int = 20,
        width: int = 150,
        fontfamily: str = "Arial",
        fontsize: int = 12,
) -> JupyterDash:
    """
    A function that takes a plotly figure and a dataframe with molecular SMILES
    and returns a dash app that dynamically generates an image of molecules in the hover box
    when hovering the mouse over datapoints.
    ...

    Attributes
    ----------
    fig : plotly.graph_objects.Figure object
        a plotly figure object containing datapoints plotted from df.
    df : pandas.DataFrame object
        a pandas dataframe that contains the data plotted in fig.
    smiles_col : str | list[str], optional
        name of the column in df containing the smiles plotted in fig (default 'SMILES').
        If provided as a list, will add a slider to choose which column is used for rendering the structures.
    show_img : bool, optional
        whether or not to generate the molecule image in the dash app (default True).
    svg_size : float, optional
        the size in pixels of the molecule drawing (default 200).
    alpha : float, optional
        the transparency of the hoverbox, 0 for full transparency 1 for full opaqueness (default 0.7).
    mol_alpha : float, optional
        the transparency of the SVG molecule image, 0 for full transparency 1 for full opaqueness (default 0.7).
    title_col : str, optional
        name of the column in df to be used as the title entry in the hover box (default None).
    show_coords : bool, optional
        whether or not to show the coordinates of the data point in the hover box (default True).
    caption_cols : list[str], optional
        list of column names in df to be included in the hover box (default None).
    caption_transform : dict[str, callable], optional
        Functions applied to specific items in all cells. The dict must follow a key: function structure where
        the key must correspond to one of the columns in subset or tooltip (default {}).
    color_col : str, optional
        name of the column in df that is used to color the datapoints in df - necessary when there is discrete conditional coloring (default None).
    marker_col : str, optional
        name of the column in df that is used to determine the marker shape of the datapoints in df (default None).
    wrap : bool, optional
        whether or not to wrap the title text to multiple lines if the length of the text is too long (default True).
    wraplen : int, optional
        the threshold length of the title text before wrapping begins - adjust when changing the width of the hover box (default 20).
    width : int, optional
        the width in pixels of the hover box (default 150).
    fontfamily : str, optional
        the font family used in the hover box (default 'Arial').
    fontsize : int, optional
        the font size used in the hover box - the font of the title line is fontsize+2 (default 12).
    """
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    df_data = df.copy()
    if color_col is not None:
        df_data[color_col] = df_data[color_col].astype(str)
    if marker_col is not None:
        df_data[marker_col] = df_data[marker_col].astype(str)

    if len(fig.data) != 1:
        colors = {index: x.marker["color"] for index, x in enumerate(fig.data)}
        if color_col is None and marker_col is None:
            raise ValueError(
                "More than one plotly curve in figure - color_col and/or marker_col needs to be specified."
            )
        if color_col is None:
            _, curve_dict = find_grouping(fig, df_data, [marker_col])
        elif marker_col is None:
            _, curve_dict = find_grouping(fig, df_data, [color_col])
        else:
            _, curve_dict = find_grouping(fig, df_data, [color_col, marker_col])
    else:
        colors = {0: "black"}

    app = JupyterDash(__name__)
    if isinstance(smiles_col, str):
        smiles_col = [smiles_col]

    if len(smiles_col) > 1:
        menu = dcc.Dropdown(
            options=[{"label": x, "value": x} for x in smiles_col],
            value=smiles_col[0],
            multi=True,
            id="smiles-menu",
            placeholder="Select a SMILES column to display",
        )
    else:
        menu = dcc.Store(id="smiles-menu", data=0)
    app.layout = html.Div(
        [
            menu,
            dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(
                id="graph-tooltip", background_color=f"rgba(255,255,255,{alpha})"
            ),
        ]
    )

    @app.callback(
        output=[
            Output("graph-tooltip", "show"),
            Output("graph-tooltip", "bbox"),
            Output("graph-tooltip", "children"),
        ],
        inputs=[Input("graph-basic-2", "hoverData"), Input("smiles-menu", "value")],
    )
    def display_hover(hoverData, value):
        if hoverData is None:
            return False, no_update, no_update

        if value is None:
            value = smiles_col
        if isinstance(value, str):
            chosen_smiles = [value]
        else:
            chosen_smiles = value

        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]
        curve_num = pt["curveNumber"]

        if len(fig.data) != 1:
            df_curve = curve_dict[curve_num].reset_index(drop=True)
            df_row = df_curve.iloc[num]
        else:
            df_row = df.iloc[num]

        hoverbox_elements = []

        if show_img:
            # The 2D image of the molecule is generated here
            for col in chosen_smiles:
                smiles = df_row[col]
                buffered = BytesIO()
                d2d = rdMolDraw2D.MolDraw2DSVG(svg_size, svg_size)
                opts = d2d.drawOptions()
                opts.clearBackground = False
                d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
                d2d.FinishDrawing()
                img_str = d2d.GetDrawingText()
                buffered.write(str.encode(img_str))
                img_str = base64.b64encode(buffered.getvalue())
                img_str = f"data:image/svg+xml;base64,{repr(img_str)[2:-1]}"
                # img_str = df_data.query(f"{col} == @smiles")[f"{col}_img"].values[0]

                if len(smiles_col) > 1:
                    hoverbox_elements.append(
                        html.H2(
                            f"{col}",
                            style={
                                "color": colors[curve_num],
                                "font-family": fontfamily,
                                "fontSize": fontsize + 2,
                            },
                        )
                    )
                hoverbox_elements.append(
                    html.Img(
                        src=img_str,
                        style={
                            "width": "100%",
                            "background-color": f"rgba(255,255,255,{mol_alpha})",
                        },
                    )
                )

        if title_col is not None:
            title = df_row[title_col]
            if len(title) > wraplen:
                if wrap:
                    title = textwrap.fill(title, width=wraplen)
                else:
                    title = title[:wraplen] + "..."

            # TODO colorbar color titles
            hoverbox_elements.append(
                html.H4(
                    f"{title}",
                    style={
                        "color": colors[curve_num],
                        "font-family": fontfamily,
                        "fontSize": fontsize,
                    },
                )
            )
        if show_coords:
            x_label = fig.layout.xaxis.title.text
            y_label = fig.layout.yaxis.title.text
            if x_label in caption_transform:
                style_str = caption_transform[x_label](pt["x"])
                hoverbox_elements.append(
                    html.P(
                        f"{x_label} : {style_str}",
                        style={
                            "color": "black",
                            "font-family": fontfamily,
                            "fontSize": fontsize,
                        },
                    )
                )
            else:
                hoverbox_elements.append(
                    html.P(
                        f"{x_label}: {pt['x']}",
                        style={
                            "color": "black",
                            "font-family": fontfamily,
                            "fontSize": fontsize,
                        },
                    )
                )
            if y_label in caption_transform:
                style_str = caption_transform[y_label](pt["y"])
                hoverbox_elements.append(
                    html.P(
                        f"{y_label} : {style_str}",
                        style={
                            "color": "black",
                            "font-family": fontfamily,
                            "fontSize": fontsize,
                        },
                    )
                )
            else:
                hoverbox_elements.append(
                    html.P(
                        f"{y_label} : {pt['y']}",
                        style={
                            "color": "black",
                            "font-family": fontfamily,
                            "fontSize": fontsize,
                        },
                    )
                )
        if caption_cols is not None:
            for caption in caption_cols:
                caption_val = df_row[caption]
                if caption in caption_transform:
                    style_str = caption_transform[caption](caption_val)
                    hoverbox_elements.append(
                        html.P(
                            f"{caption} : {style_str}",
                            style={
                                "color": "black",
                                "font-family": fontfamily,
                                "fontSize": fontsize,
                            },
                        )
                    )
                else:
                    hoverbox_elements.append(
                        html.P(
                            f"{caption} : {caption_val}",
                            style={
                                "color": "black",
                                "font-family": fontfamily,
                                "fontSize": fontsize,
                            },
                        )
                    )
        children = [
            html.Div(
                hoverbox_elements,
                style={
                    "width": f"{width}px",
                    "white-space": "normal",
                },
            )
        ]

        return True, bbox, children

    return app


def smi2svg(smi):
    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    drawer = Draw.MolDraw2DSVG(200, 200)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    return svg


def smi2fp(smi):
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=1024)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def visualize_chemical_space(smiles=None, components_mode='pca', color_values=None, label=''):
    df_esol = pd.DataFrame({'smiles': smiles, 'Compound ID': np.array([f'Compound_{i}' for i in range(len(smiles))])})
    if color_values is None:
        color_values = np.arange(0, len(smiles))
    if components_mode == 'pca':
        esol_fps = np.array([smi2fp(smi) for smi in df_esol['smiles']])
        pca = PCA(n_components=2)
        components = pca.fit_transform(esol_fps.reshape(-1, 1024))
        df_esol['PCA-1'] = components[:, 0]
        df_esol['PCA-2'] = components[:, 1]
        df_esol['color'] = color_values

    elif components_mode == 'mds':
        components = transorm_MDS([Chem.MolFromSmiles(smi) for smi in df_esol['smiles']], 2)
        df_esol['PCA-1'] = components[:, 0]
        df_esol['PCA-2'] = components[:, 1]
        df_esol['color'] = color_values

    fig_pca = px.scatter(df_esol,
                         x="PCA-1",
                         y="PCA-2",
                         # color='color',
                         title='PCA of morgan fingerprints',
                         labels={'color': label},
                         width=1200,
                         height=800)

    app_pca = add_molecules(fig=fig_pca,
                            df=df_esol,
                            smiles_col='smiles',
                            title_col='Compound ID',
                            )
    return app_pca, fig_pca


def online_chemical_space(components_mode='pca', color_values=None, label='', run_id=0, sampler_id=0):
    r = redis.Redis(db=run_id)
    num_ligands = int(r.get('num_ligands'))
    ligands: List[LigandInfo] = [dict(r.get(i)) for i in range(num_ligands) if r.get(i) is not None]
    smiles = [lig.smiles for lig in ligands]
    dg = [lig.dg if (lig.dg is not None) and lig.run_id == sampler_id else 0. for lig in ligands]

    df_esol = pd.DataFrame({'smiles': smiles, 'Compound ID': np.array([f'Compound_{i}' for i in range(len(smiles))])})
    if color_values is None:
        color_values = dg
    if components_mode == 'pca':
        esol_fps = np.array([smi2fp(smi) for smi in df_esol['smiles']])
        pca = PCA(n_components=2)
        components = pca.fit_transform(esol_fps.reshape(-1, 1024))
        df_esol['PCA-1'] = components[:, 0]
        df_esol['PCA-2'] = components[:, 1]
        df_esol['color'] = color_values

    elif components_mode == 'mds':
        components = transorm_MDS([Chem.MolFromSmiles(smi) for smi in df_esol['smiles']], 2)
        df_esol['PCA-1'] = components[:, 0]
        df_esol['PCA-2'] = components[:, 1]
        df_esol['color'] = color_values

    fig_pca = px.scatter(df_esol,
                         x="PCA-1",
                         y="PCA-2",
                         color='color',
                         title='PCA of morgan fingerprints',
                         labels={'color': label},
                         width=1200,
                         height=800)

    # fig_pca.add_trace(go.Scatter(x=df_esol['PCA-1'], y=df_esol['PCA-2'],
    #                             mode='lines',
    #                             name='trajectory'))

    app_pca = add_molecules(fig=fig_pca,
                            df=df_esol,
                            smiles_col='smiles',
                            title_col='Compound ID',
                            )
    return app_pca, fig_pca


def visualize_3d_mol(mol: Union[str, AllChem.Mol]):
    if isinstance(mol, str):
        mol = AllChem.MolFromSmiles(mol)

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    xyzview = py3Dmol.view()  # (width=400,height=400)
    xyzview.addModel(mblock, 'mol')
    xyzview.setStyle({'stick': {}})
    xyzview.setBackgroundColor('white')
    xyzview.zoomTo()
    return xyzview


def create_mol3d_style(
        atoms_list: List[List[dict]],
        visualization_type: List[str] = None,
        color_elements: List[str] = None,
        color_scheme=None,
        default_c=None
):
    """Function to create styles input for Molecule3dViewer

    @param atoms_list
    A list of atoms. Each atom should be a dict with keys: 'name', 'residue_name', 'chain'
    @param visualization_type
    A type of molecule visualization graphic: 'stick' | 'cartoon' | 'sphere'.
    @param color_elements
    Elements to apply color scheme to: 'atom' | 'residue' | 'residue_type' | 'chain'.
    @param color_scheme
    Color scheme used to style moleule elements.
    This should be a dict with keys being names of atoms, residues, residue types or chains,
    depending on the value of color_element argument. If no value is provided, default color
    schemes will be used.
    @param default_c
    """

    if default_c is None:
        default_c = ["#f00000"] * len(atoms_list)
    if visualization_type is None:
        visualization_type = ["stick"] * len(atoms_list)
    if color_elements is None:
        color_elements = ["atom"] * len(atoms_list)

    atom_styles = []

    for i, atoms in enumerate(atoms_list):
        if visualization_type[i] not in ['stick', 'cartoon', 'sphere']:
            raise Exception("Invalid argument type: visualization_type.\
        Should be: 'stick' | 'cartoon' | 'sphere'.")

        if color_elements[i] not in ['atom', 'residue', 'residue_type', 'chain']:
            raise Exception("Invalid argument type: color_element.\
        Should be: 'atom' | 'residue' | 'residue_type' | 'chain'.")

        if not isinstance(atoms_list[0], list):
            raise Exception("Invalid argument type: atoms. Should be a list of dict.")

        if color_scheme and not isinstance(color_scheme, dict):
            raise Exception("Invalid argument type: color_scheme. Should be a dict.")

        default_color = '#ABABAB'
        color_element = color_elements[i]

        if color_scheme is None:
            color_scheme = {
                'atom': ATOM_COLORS,
                'residue': RESIDUE_COLORS,
                'residue_type': RESIDUE_TYPE_COLORS,
                'chain': CHAIN_COLORS
            }[color_element]

        if color_element == 'residue_type':
            residue_type_colors_map = {}
            for aa_class_name, aa_class_members in AMINO_ACID_CLASSES.items():
                for aa in aa_class_members:
                    residue_type_colors_map[aa] = color_scheme.get(aa_class_name, default_color)
            color_scheme = residue_type_colors_map

        for a in atoms:
            if color_element == 'atom':
                atom_color = color_scheme.get(a['name'], default_color)
            if color_element in ['residue', 'residue_type']:
                atom_color = color_scheme.get(a['residue_name'], default_color)
            if color_element == 'chain':
                atom_color = color_scheme.get(a['chain'], default_color)

            atom_styles.append({
                'visualization_type': visualization_type[i],
                'color': atom_color if default_c[i] is None else default_c[i]
            })

    return atom_styles


def plot_dg(run_id=0):
    r = redis.Redis(db=run_id)
    current_step = r.get('computed_step')
    computed_steps_ids = json.loads(r.get('computed_steps_ids'))
    chemical_space = {}
    for k in r.keys():
        if k != 'computed_step' and k != 'computed_steps_ids':
            lig = json.loads(r.get(k))
            if isinstance(lig, dict):
                chemical_space[lig['compound_id']] = lig

    steps = []
    min_dg = []
    min_smiles = []
    min_ids = []
    all_mols = []
    for i in range(len(computed_steps_ids)):
        steps.append(float(i))
        mols = [chemical_space[k] for k in computed_steps_ids[i]]
        all_mols += mols
        min_id = np.argmin([mol['score'] for mol in all_mols])
        min_dg.append(float(all_mols[min_id]['score']))
        min_ids.append(all_mols[min_id]['compound_id'])
        min_smiles.append(all_mols[min_id]['smiles'])

    df = pd.DataFrame({'x': steps, 'y': min_dg, 'smiles': min_smiles, 'id': min_ids})
    print(df)
    fig = px.line(data_frame=df, x='x', y='y', hover_name='id', hover_data=["smiles"], markers=True)
    return fig


def visualize_sampler_run(run_id=0):
    app = JupyterDash(__name__)
    app.layout = html.Div([
        html.H4(f'visualization progresses run: {run_id}'),
        dcc.Graph(id="graph"),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        )
    ])

    @app.callback(Output('graph', 'figure'),
                  Input('interval-component', 'n_intervals'))
    def update_metrics(n):
        return plot_dg(run_id)

    return app


def visualize_3d_complex(ligand_pdb, protein_pdb, range_within=None,
                         protein_rep='cartoon',
                         ligand_rep='stick',
                         pocket_rep='stick',
                         protein_color_scheme='residue',
                         range_color_scheme='residue'):
    from prody import parsePDB, writePDB, calcCenter

    def merge_datas(data_a, data_b):
        result = copy.deepcopy(data_a)
        max_residue = np.max([data_a['atoms'][i]['residue_index'] for i in range(len(data_a['atoms']))]) + 1
        max_atom_id = np.max([data_a['atoms'][i]['serial'] for i in range(len(data_a['atoms']))]) + 1
        for atom in data_b['atoms']:
            new_data = copy.deepcopy(atom)
            new_data['residue_index'] += max_residue
            new_data['serial'] += max_atom_id
            result['atoms'].append(new_data)

        for atom in data_b['bonds']:
            new_data = copy.deepcopy(atom)
            new_data['atom1_index'] += max_atom_id
            new_data['atom2_index'] += max_atom_id
            result['bonds'].append(new_data)

        return result

    def select_pdb(pdb_file, selection_string, name='temp.pdb'):
        pdb = parsePDB(pdb_file).select(selection_string)
        writePDB(osp.join('./tmp', name), pdb)
        return osp.join('./tmp', name)

    def select_binding_pocket(range_selection, name='temp.pdb'):
        pdb_ligand = parsePDB(ligand_pdb)
        center = calcCenter(pdb_ligand)
        selection_string = f'same residue as (within {range_selection} of center) and noh'
        pdb = parsePDB(protein_pdb).select(selection_string, center=center)
        writePDB(osp.join('./tmp', name), pdb)
        return osp.join('./tmp', name)

    def read_remark(file_path):
        res = ''
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('REMARK'):
                    res += line + '\n'

        return res

    protein_parser = PdbParser(select_pdb(protein_pdb, 'protein and noh', name='protein.pdb'))
    ligand_parser = PdbParser(select_pdb(ligand_pdb, 'not protein and noh', name='ligand.pdb'))
    if range_within is not None:
        poket_parser = PdbParser(select_binding_pocket(range_within, name='poket.pdb'))
        data_poket = poket_parser.mol3d_data()
    else:
        poket_parser = None
        data_poket = None

    data_protein = protein_parser.mol3d_data()
    data_ligand = ligand_parser.mol3d_data()

    data = merge_datas(data_protein, data_ligand)
    if data_poket is not None:
        data = merge_datas(data, data_poket)

    style = create_mol3d_style([data_protein['atoms'], data_ligand['atoms'], data_poket['atoms']] if data_poket is not None else
                               [data_protein['atoms'], data_ligand['atoms']],
                               [protein_rep, ligand_rep, pocket_rep],
                               color_elements=[protein_color_scheme, 'atom', range_color_scheme],
                               default_c=[None, None, None])

    return data, style, data_protein, data_ligand, data_poket, read_remark(ligand_pdb)


def visualize_3d_complex_app(ligand_pdb, protein_pdb, range_within=None):
    app = JupyterDash(__name__)
    data, style, data_protein, data_ligand, data_poket, head = visualize_3d_complex(ligand_pdb, protein_pdb, range_within)

    app.layout = html.Div([
        html.Div([
            dashbio.Molecule3dViewer(
                id='dashbio-default-molecule3d',
                modelData=data,
                styles=style
            ),

        ], style={'padding': 10, 'flex': 1}),
        html.Div([

            dcc.Dropdown(
                id="dashbio-protein_color_scheme",
                value='residue',
                searchable=False,
                placeholder="Protein color scheme",
                options=[{"label": s.replace('_', ' ').capitalize(), "value": s} for s in ["residue", "atom", "residue_type", "chain"]]
            ),
            html.Hr(),
            dcc.Dropdown(
                id="dashbio-protein_rep",
                value='cartoon',
                searchable=False,
                placeholder="Protein Style",
                # style={'color': 'gray', 'background': 'gray'},
                options=[{"label": s.replace('_', ' ').capitalize(), "value": s} for s in ['stick', 'cartoon', 'sphere']]
            ),

            dcc.Dropdown(
                id="dashbio-ligand_rep",
                value='stick',
                searchable=False,
                placeholder="Ligand Style",
                options=[{"label": s.replace('_', ' ').capitalize(), "value": s} for s in ['stick', 'cartoon', 'sphere']]
            ),

            dcc.Dropdown(
                id="dashbio-pocket_rep",
                value='stick',
                searchable=False,
                placeholder="Pocket Style",
                options=[{"label": s.replace('_', ' ').capitalize(), "value": s} for s in ['stick', 'cartoon', 'sphere']]
            ),
            html.Hr(),
            dcc.Slider(
                id="zoomfactor-slider",
                min=0.4,
                max=2.0,
                step=None,
                marks={0.4: "0.4", 0.8: "0.8", 1.2: "1.2", 1.6: "1.6", 2.0: "2.0"},
                value=0.8,
            ),
            dcc.Slider(
                id="range-slider",
                min=0,
                max=15,
                step=1,
                marks={i: str(i) for i in range(0, 16, 5)},
                value=10,
            ),
            html.Hr(),
            dcc.Markdown(head, style={'color': '#b4b4b4'}),
            html.Hr(),
            dcc.Markdown("Selection data", style={'color': '#b4b4b4'}),
            html.Hr(),
            html.Div(id='default-molecule3d-output')

        ], style={'padding': 10, 'flex': 1})
    ], style={'display': 'flex', 'flex-direction': 'row'})

    @app.callback(
        Output('default-molecule3d-output', 'children'),
        Output("dashbio-default-molecule3d", "zoom"),
        Output('dashbio-default-molecule3d', 'modelData'),
        Output('dashbio-default-molecule3d', 'styles'),
        Input('dashbio-default-molecule3d', 'selectedAtomIds'),
        Input("dashbio-protein_color_scheme", "value"),
        Input("dashbio-protein_rep", "value"),
        Input("dashbio-ligand_rep", "value"),
        Input("dashbio-pocket_rep", "value"),
        Input("zoomfactor-slider", "value"),
        Input("range-slider", "value"),
    )
    def show_selected_atoms(atom_ids,
                            protein_color_scheme,
                            protein_rep,
                            ligand_rep,
                            pocket_rep,
                            slider,
                            range_w
                            ):
        if range_w == 0:
            range_w = None
        data, style, data_protein, data_ligand, data_poket, _ = visualize_3d_complex(ligand_pdb, protein_pdb, range_w)

        style = create_mol3d_style([data_protein['atoms'], data_ligand['atoms'], data_poket['atoms']] if data_poket is not None else
                                   [data_protein['atoms'], data_ligand['atoms']],
                                   [protein_rep, ligand_rep, pocket_rep],
                                   color_elements=[protein_color_scheme, 'atom', protein_color_scheme],
                                   default_c=[None, None, None])

        if atom_ids is None or len(atom_ids) == 0:
            layout = 'No atom has been selected. Click somewhere on the molecular \
            structure to select an atom.'
        else:
            layout = [html.Div([
                dcc.Markdown('Element: {}, '.format(data['atoms'][atm]['elem']) +
                             'Chain: {}, '.format(data['atoms'][atm]['chain']) +
                             'Residue name: {} '.format(data['atoms'][atm]['residue_name']), style={'color': '#b4b4b4'}),
            ]) for atm in atom_ids]

        return layout, {"factor": slider, "animationDuration": 10, "fixedPath": False}, data, style

    return app


def run_main_visualization_app():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    alpha = 0.7
    app.layout = html.Div([
        html.H1('Drug Discovery Supervisor'),
        dcc.Interval(
            id='interval-component',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        ),
        dcc.Tabs(id="tabs-system", value='tab-1', children=[
            dcc.Tab(label='Chemical Space',
                    value='tab-chemical-space',
                    children=html.Div(className='control-tab',
                                      children=[
                                          dcc.Store(id="smiles-menu", data=0),
                                          dcc.Graph(id="graph-basic-2", clear_on_unhover=True),
                                          dcc.Tooltip(
                                              id="graph-tooltip", background_color=f"rgba(255,255,255,{alpha})"
                                          ),
                                      ]
                                      )
                    ),
            dcc.Tab(label='Optimization Progresses',
                    value='tab-optimization',
                    children=html.Div(className='control-tab',
                                      children=[
                                          dcc.Graph(id="graph_min_dg"),
                                      ]
                                      )
                    ),
            dcc.Tab(label='Docking Visualization',
                    value='tab-docking-visualization',
                    children=html.Div(className='control-tab',
                                      children=[
                                          dcc.Graph(id="graph_min_dg"),
                                      ]
                                      )
                    ),
            dcc.Tab(label='Run Initialization',
                    value='tab-run-initialization',
                    children=html.Div(className='control-tab',
                                      children=[
                                          dcc.Graph(id="graph_min_dg"),
                                      ]
                                      )
                    ),
        ]),
        html.Div(id='tabs-content-example-graph')
    ])
