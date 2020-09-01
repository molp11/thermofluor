# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 17:12:32 2018

@author: molp
"""

from functools import lru_cache

from os import listdir
from os.path import dirname, join
import pickle

import re
import numpy as np
import pandas as pd
from lmfit import Model

from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox, layout
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Div, Legend, TapTool
from bokeh.models.widgets import DataTable, Select, TableColumn, Button, TextInput, RadioButtonGroup, HTMLTemplateFormatter, CheckboxGroup
from bokeh.plotting import figure

from io import StringIO
import base64

degree_sign= u'\N{DEGREE SIGN}'

# equations & methods for curve fitting / model selection
def monophasicBS(x, bottom, top, v50, slope):
    '''
    monophasic Boltzman Sigmoidal equation with bottom = 0
    '''
    return bottom + (top - bottom) / (1 + np.exp((v50 - x) / slope))

def biphasicBS(x, bottoma, topa, v50A, slopea, bottomb, topb, v50B, slopeb):
    '''
    biphasic Boltzman Sigmoidal equation with bottoms = 0
    '''
    return bottoma + (topa - bottoma) / (1 + np.exp((v50A - x) / slopea)) + bottomb + (topb - bottomb) / (1 + np.exp((v50B - x) / slopeb))

def model_fit(df, setting):
    '''
    select monophasic / biphasic model based on AIC
    '''
    min_signal = df[df.y == np.max(df.y)].x.values[0]
    max_signal = np.max(df.y)
    min_temp = df[df.y == np.min(df.y)].x.values[0]
    min_signal = np.min(df.y)
    if bottom_input.value == ' ':
        bottom_init = min_signal
    else:
        bottom_init = float(bottom_input.value)
    if bottoma_input.value == ' ':
        bottoma_init = min_signal
    else:
        bottoma_init = float(bottoma_input.value)
    if bottomb_input.value == ' ':
        bottomb_init = min_signal
    else:
        bottomb_init = float(bottomb_input.value)
    if top_input.value == ' ':
        top_init = max_signal
    else:
        top_init = float(top_input.value)
    if topa_input.value == ' ':
        topa_init = np.max(df.y)
    else:
        topa_init = float(topa_input.value)
    if topb_input.value == ' ':
        topb_init = np.max(df.y)
    else:
        topb_init = float(topb_input.value)
    if slope_input.value == ' ':
        slope_init = ((max_signal - min_signal)/(max_temp - min_temp))
    else:
        slope_init = float(slope_input.value)
    if slopea_input.value == ' ':
        slopea_init = ((max_signal - min_signal)/(max_temp - min_temp))
    else:
        slopea_init = float(slopea_input.value)
    if slopeb_input.value == ' ':
        slopeb_init = ((max_signal - min_signal)/(max_temp - min_temp))
    else:
        slopeb_init = float(slopeb_input.value)
    if v50_input.value == ' ':
        v50_init = ((max_temp - min_temp)/2)
    else:
        v50_init = float(v50_input.value)
    if v50a_input.value == ' ':
        v50a_init = ((max_temp - min_temp)/2)
    else:
        v50a_init = float(v50a_input.value)
    if v50b_input.value == ' ':
        v50b_init = ((max_temp - min_temp)/2)
    else:
        v50b_init = float(v50b_input.value)
    x, y = df.x, df.y
    if setting == 'mono':
        # monophasic model
        mmodel = Model(monophasicBS)
        mresult = mmodel.fit(y, x=x, bottom=bottom_init, top=top_init, v50=v50_init,
                     slope=slope_init)
        best_fit = mresult
    elif setting == 'bi':
        # biphasic model
        bmodel = Model(biphasicBS)
        bresult = bmodel.fit(y, x=x, bottoma=bottoma_init, topa=topa_init, v50A=v50a_init,
                             slopea=slopea_init, bottomb=bottomb_init, topb=topb_init,
                             v50B=v50b_init, slopeb=slopeb_init)
        best_fit = bresult

    return best_fit

# bokeh methods
def heatmap_select(attr, old, new):
    global sample
    try:
        selected_index = heatmap_source.selected["1d"]["indices"][0]
        heatmap_row.value = str(selected_index)
        heatmap_cell_column_1.value = str(heatmap_source.data["sample"][selected_index])
        sample = heatmap_cell_column_1.value
    except IndexError:
        pass
    update()

def table_select(attr, old, new):
    global sample
    try:
        selected_index = source_data_table.selected["1d"]["indices"][0]
        table_row.value = str(selected_index)
        table_cell_column_1.value = str(source_data_table.data["Sample"][selected_index])
        sample = table_cell_column_1.value
    except IndexError:
        pass
    update()

def selection_range(attr, old, new):
    selected_range = source_model.selected.indices
    global select_db
    select_db[sample] = selected_range
    update()

def source_update(attr, old, new):
    update()

def radio_update(attr, old, new):
    update(full=False)

def checkbox_update(attr, old, new):
    global state
    if 0 in checkbox_group.active and 1 in checkbox_group.active:
        state = 2
    elif 0 in checkbox_group.active and 1 not in checkbox_group.active:
        state = 0
    elif 0 not in checkbox_group.active and 1 in checkbox_group.active:
        state = 1
    update()

def file_callback(attr, old, new):
    global filename
    filename=file_source.data['file_name']
    global output_filename
    output_filename = filename[0].strip('.csv')+'-out.csv'
    raw_contents = file_source.data['file_contents'][0]
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_contents = file_contents.decode("utf-8-sig")
    file_io = StringIO(file_contents)
    global df
    df = pd.read_csv(file_io)
    update_dataset()

def update_dataset():
    tmp_sample = list(df)[1] # automatically show 1st sample

    # make sure all column names are interpreted as strings
    sample_names = list(df)
    string_names = [str(i) for i in sample_names]
    df.columns = string_names

    # convert to dictionary to accomodate variable length series for individual samples
    global select_db
    select_db = {}
    global db
    db = {}
    for s in list(df)[1:]: # first column must be Temperature independent variable
        tmpdf = df[[list(df)[0], s]]
        tmpdf.columns = ['x', 'y']
        tmpdf = tmpdf.dropna()
        db[s] = tmpdf

    # calculate fits for all samples
    global melt_temps
    melt_temps = []
    for s in list(df)[1:]:
        model.title.text = 'Fitting Sample: '+str(s)+'...'
        tmp = ['-', '-', '-', '-', '-'] # current row

        sample_df = db[s]
        source_model_line_m.data = source_model_line_m.from_df(pd.DataFrame(data={ 'x' : [], 'y' : []}))
        source_model_line_b.data = source_model_line_b.from_df(pd.DataFrame(data={ 'x' : [], 'y' : []}))
        source_model.data = source_model.from_df(sample_df)

        # only fit points after 1st minimum and before maximum
        max_temp = sample_df[sample_df.y == np.max(sample_df.y)].x.values[0]
        tmp_df = sample_df[sample_df.x <= max_temp]
        min_temp = sample_df[sample_df.y == np.min(sample_df.y)].x.values[0]
        tmp_df = tmp_df[tmp_df.x >= min_temp]

        # fit both models
        try:
            mfit = model_fit(tmp_df, 'mono')

            source_model_line_m.data = source_model_line_m.from_df(pd.DataFrame(data={ 'x' : tmp_df.x, 'y' : mfit.best_fit}))
            try:
                tmp[0] = mfit.best_values['v50']
                tmp[3] = mfit.aic
            except:
                pass

            if 1 in checkbox_group.active:
                bfit = model_fit(tmp_df, 'bi')

                source_model_line_b.data = source_model_line_b.from_df(pd.DataFrame(data={ 'x' : tmp_df.x, 'y' : bfit.best_fit}))
                try:
                    # set closest biphasic Tm to monophasic Tm as biphasic Tm-1
                    tmp_temps = [tmp] + melt_temps # to avoid taking mean of empty melt_temps column
                    mean_temp = np.mean([i[0] for i in tmp_temps])
                    if np.abs(bfit.best_values['v50A'] - mean_temp) <= np.abs(bfit.best_values['v50B'] - mean_temp):
                        tmp[1] = bfit.best_values['v50A']
                        tmp[2] = bfit.best_values['v50B']
                    else:
                        tmp[2] = bfit.best_values['v50A']
                        tmp[1] = bfit.best_values['v50B']
                    tmp[4] = bfit.aic
                except:
                    pass
        except:
            model.title.text = 'Fitting Error: '+str(s)+'...'
        melt_temps.append(tmp)

    global melt_df
    melt_df = pd.DataFrame(melt_temps, columns=['v50', 'v50A', 'v50B', 'aicA', 'aicB'])

    # data source
    color = []
    alpha = []
    color1 = []
    alpha1 = []
    color2 = []
    alpha2 = []
    numbers = []
    numbers1 = []
    numbers2 = []
    probability = []

    for mt in range(len(melt_df.v50)):
        if melt_df.v50[mt] != '-':
            numbers.append(melt_df.v50[mt])
            mean_temp = np.mean(numbers)
            max_temp = np.max(numbers)
            min_temp = np.min(numbers)
        if melt_df.v50A[mt] != '-':
            numbers1.append(melt_df.v50A[mt])
            numbers2.append(melt_df.v50B[mt])
            mean_temp1 = np.mean(numbers1)
            max_temp1 = np.max(numbers1)
            min_temp1 = np.min(numbers1)
            mean_temp2 = np.mean(numbers2)
            max_temp2 = np.max(numbers2)
            min_temp2 = np.min(numbers2)

    for mt in range(len(melt_df.v50)):
        if melt_df.v50[mt] == '-':
            color.append("lightgrey")
            alpha.append(0.6)
        else:
            if melt_df.v50[mt] > mean_temp:
                color.append("#e31a1c")#"#fb9a99")
                alpha.append((melt_df.v50[mt] - mean_temp) / (max_temp - mean_temp) * 0.9 + 0.05)
            if melt_df.v50[mt] <= mean_temp:
                color.append("#1f78b4")#"#a6cee3")
                alpha.append((1 - (melt_df.v50[mt] - min_temp) / (mean_temp - min_temp)) * 0.9 + 0.05)
        if melt_df.v50A[mt] == '-':
            color1.append("lightgrey")
            alpha1.append(0.6)
            color2.append("lightgrey")
            alpha2.append(0.6)
        else:
            if melt_df.v50A[mt] > mean_temp:
                color1.append("#e31a1c")#"#fb9a99")
                alpha1.append((melt_df.v50A[mt] - mean_temp) / (max_temp - mean_temp) * 0.9 + 0.05)
            if melt_df.v50A[mt] <= mean_temp:
                color1.append("#1f78b4")#"#a6cee3")
                alpha1.append((1 - (melt_df.v50A[mt] - min_temp) / (mean_temp - min_temp)) * 0.9 + 0.05)
            if melt_df.v50B[mt] > mean_temp:
                color2.append("#e31a1c")#"#fb9a99")
                alpha2.append((melt_df.v50B[mt] - mean_temp) / (max_temp - mean_temp) * 0.9 + 0.05)
            if melt_df.v50B[mt] <= mean_temp:
                color2.append("#1f78b4")#"#a6cee3")
                alpha2.append((1 - (melt_df.v50B[mt] - min_temp) / (mean_temp - min_temp)) * 0.9 + 0.05)
        if melt_df.aicA[mt] != '-' and melt_df.aicB[mt] != '-':
            probability.append(np.exp(-1*(melt_df.aicA[mt]-melt_df.aicB[mt])/2))
        else:
            probability.append('-')

    data=dict(
        xname=xname[:len(color)],
        yname=yname[:len(color)],
        colors=color,
        alphas=alpha,
        colors1=color1,
        alphas1=alpha1,
        colors2=color2,
        alphas2=alpha2,
        v50=melt_df.v50,
        v50A=melt_df.v50A,
        v50B=melt_df.v50B,
        aicA=melt_df.aicA,
        aicB=melt_df.aicB,
        sample=list(df)[1:],
        probability=probability
    )
    source.data = source.from_df(pd.DataFrame(data))
    sample_ix = list(df)[1:].index(tmp_sample)
    model.title.text = tmp_sample+':'
    if 0 in checkbox_group.active:
        model.title.text = model.title.text+' Tm = '+str(source.data['v50'][sample_ix])
    if 1 in checkbox_group.active:
        model.title.text = model.title.text+' Tm-1 = '+str(source.data['v50A'][sample_ix])+' Tm-2 = '+str(source.data['v50B'][sample_ix])
    output_filename = filename[0].strip('.csv')+'-out.csv'
    download_button.callback.args = dict(source=source_data_table, file_name=output_filename, state=state)
    copy_button.callback.args = dict(source=source_data_table, state=state)
    global sample
    sample = str(list(df)[1])
    update()

def update(full=True):
    sample_ix = list(df)[1:].index(sample) # get sample index
    sample_df = db[sample]
    if sample in list(select_db): # if a selection range has been made for this sample
        selected_range = select_db[sample]
        tmp_df = db[sample].iloc[selected_range, :].reset_index(drop=True)
        tmp_df = tmp_df.sort_values('x')
    else:
        max_temp = sample_df[sample_df.y == np.max(sample_df.y)].x.values[0]
        tmp_df = sample_df[sample_df.x <= max_temp]
        min_temp = sample_df[sample_df.y == np.min(sample_df.y)].x.values[0]
        tmp_df = tmp_df[tmp_df.x >= min_temp]
    if full == True:
        source_model.data = source_model.from_df(sample_df)
        source_model_line_m.data = source_model_line_m.from_df(pd.DataFrame(data=dict(x=[], y=[])))
        source_model_line_b.data = source_model_line_b.from_df(pd.DataFrame(data=dict(x=[], y=[])))

        global melt_df
        if 0 in checkbox_group.active:
            hold = model.title.text
            model.title.text = 'Fitting Sample: '+str(sample)+'...'
            new_model = model_fit(tmp_df, 'mono')
            model.title.text = hold

            melt_df.at[sample_ix, 'v50'] = new_model.best_values['v50']
            melt_df.at[sample_ix, 'aicA'] = new_model.aic
            source_model_line_m.data = source_model_line_m.from_df(pd.DataFrame(data={ 'x' : tmp_df.x, 'y' : new_model.best_fit}))
        if 1 in checkbox_group.active:
            hold = model.title.text
            model.title.text = 'Fitting Sample: '+str(sample)+'...'
            new_model = model_fit(tmp_df, 'bi')
            model.title.text = hold

            if new_model.best_values['v50A'] <= new_model.best_values['v50B']:
                melt_df.at[sample_ix, 'v50A'] = new_model.best_values['v50A']
                melt_df.at[sample_ix, 'v50B'] = new_model.best_values['v50B']
            else:
                melt_df.at[sample_ix, 'v50A'] = new_model.best_values['v50B']
                melt_df.at[sample_ix, 'v50B'] = new_model.best_values['v50A']
            melt_df.at[sample_ix, 'aicB'] = new_model.aic
            source_model_line_b.data = source_model_line_b.from_df(pd.DataFrame(data={ 'x' : tmp_df.x, 'y' : new_model.best_fit}))

    # data source
    color = []
    alpha = []
    color1 = []
    alpha1 = []
    color2 = []
    alpha2 = []
    numbers = []
    numbers1 = []
    numbers2 = []
    probability = []

    for mt in range(len(melt_df.v50)):
        if melt_df.v50[mt] != '-':
            numbers.append(melt_df.v50[mt])
            mean_temp = np.mean(numbers)
            max_temp = np.max(numbers)
            min_temp = np.min(numbers)
        if melt_df.v50A[mt] != '-':
            numbers1.append(melt_df.v50A[mt])
            numbers2.append(melt_df.v50B[mt])
            mean_temp1 = np.mean(numbers1)
            max_temp1 = np.max(numbers1)
            min_temp1 = np.min(numbers1)
            mean_temp2 = np.mean(numbers2)
            max_temp2 = np.max(numbers2)
            min_temp2 = np.min(numbers2)

    for mt in range(len(melt_df.v50)):
        if melt_df.v50[mt] == '-':
            color.append("lightgrey")
            alpha.append(0.6)
        else:
            if melt_df.v50[mt] > mean_temp:
                color.append("#e31a1c")#"#fb9a99")
                alpha.append((melt_df.v50[mt] - mean_temp) / (max_temp - mean_temp) * 0.9 + 0.05)
            if melt_df.v50[mt] <= mean_temp:
                color.append("#1f78b4")#"#a6cee3")
                alpha.append((1 - (melt_df.v50[mt] - min_temp) / (mean_temp - min_temp)) * 0.9 + 0.05)
        if melt_df.v50A[mt] == '-':
            color1.append("lightgrey")
            alpha1.append(0.6)
            color2.append("lightgrey")
            alpha2.append(0.6)
        else:
            if melt_df.v50A[mt] > mean_temp:
                color1.append("#e31a1c")#"#fb9a99")
                alpha1.append((melt_df.v50A[mt] - mean_temp) / (max_temp - mean_temp) * 0.9 + 0.05)
            if melt_df.v50A[mt] <= mean_temp:
                color1.append("#1f78b4")#"#a6cee3")
                alpha1.append((1 - (melt_df.v50A[mt] - min_temp) / (mean_temp - min_temp)) * 0.9 + 0.05)
            if melt_df.v50B[mt] > mean_temp:
                color2.append("#e31a1c")#"#fb9a99")
                alpha2.append((melt_df.v50B[mt] - mean_temp) / (max_temp - mean_temp) * 0.9 + 0.05)
            if melt_df.v50B[mt] <= mean_temp:
                color2.append("#1f78b4")#"#a6cee3")
                alpha2.append((1 - (melt_df.v50B[mt] - min_temp) / (mean_temp - min_temp)) * 0.9 + 0.05)
        if melt_df.aicA[mt] != '-' and melt_df.aicB[mt] != '-':
            probability.append(np.exp(-1*(melt_df.aicA[mt]-melt_df.aicB[mt])/2))
        else:
            probability.append('-')

    data=dict(
        xname=xname[:len(color)],
        yname=yname[:len(color)],
        colors=color,
        alphas=alpha,
        colors1=color1,
        alphas1=alpha1,
        colors2=color2,
        alphas2=alpha2,
        v50=melt_df.v50,
        v50A=melt_df.v50A,
        v50B=melt_df.v50B,
        aicA=melt_df.aicA,
        aicB=melt_df.aicB,
        sample=list(df)[1:],
        probability=probability
    )

    source.data = source.from_df(pd.DataFrame(data))
    model.title.text = sample+':'
    table_df = pd.DataFrame(data={'Sample' : source.data['sample']})
    if 0 in checkbox_group.active:
        model.title.text = model.title.text+' Tm = '+str(source.data['v50'][sample_ix])
        table_df['Tm'] = source.data['v50']
        table_df['AIC monophasic'] = source.data['aicA']
    if 1 in checkbox_group.active:
        model.title.text = model.title.text+' Tm-1 = '+str(source.data['v50A'][sample_ix])+' Tm-2 = '+str(source.data['v50B'][sample_ix])
        table_df['Tm-1'] = source.data['v50A']
        table_df['Tm-2'] = source.data['v50B']
        table_df['AIC biphasic'] = source.data['aicB']
        table_df['probability'] = source.data['probability']

    source_data_table.data = source_data_table.from_df(table_df)
    download_button.callback = CustomJS(args=dict(source=source_data_table, file_name=output_filename, state=state),
                                code=open(join(dirname(__file__), "download.js")).read())
    copy_button.callback = CustomJS(args=dict(source=source_data_table, state=state),
                            code=open(join(dirname(__file__), "copy.js")).read())

    if radio_group.active == 0:
        heatmap_df = pd.DataFrame(data={'sample' : source.data['sample'], 'xname' : source.data['xname'], 'yname' : source.data['yname'],
                        'colors' : source.data['colors'], 'alphas' : source.data['alphas'], 'v50' : source.data['v50']})
    elif radio_group.active == 1:
        heatmap_df = pd.DataFrame(data={'sample' : source.data['sample'], 'xname' : source.data['xname'], 'yname' : source.data['yname'],
                        'colors' : source.data['colors1'], 'alphas' : source.data['alphas1'], 'v50' : source.data['v50A']})
    elif radio_group.active == 2:
        heatmap_df = pd.DataFrame(data={'sample' : source.data['sample'], 'xname' : source.data['xname'], 'yname' : source.data['yname'],
                        'colors' : source.data['colors2'], 'alphas' : source.data['alphas2'], 'v50' : source.data['v50B']})

    heatmap_source.data = heatmap_source.from_df(heatmap_df)

# read data file
data_file = join(dirname(__file__), 'test_data/CS_example_Mx3005p.csv')
df = pd.read_csv(data_file)
sample = list(df)[0] # automatically show 1st sample

# make sure all column names are interpreted as strings
sample_names = list(df)
string_names = [str(i) for i in sample_names]
df.columns = string_names

# convert to dictionary to accomodate variable length series for individual samples
select_db = {}
db = {}
for sample in list(df)[1:]: # first column must be Temperature independent variable
    tmpdf = df[[list(df)[0], sample]]
    tmpdf.columns = ['x', 'y']
    db[sample] = tmpdf

# heatmap plate layouts
columns = []
for c in range(1, 13):
    columns.append(str(c))

rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
xname, yname = [], []
for r in rows:
    for c in columns:
        yname.append(r)
        xname.append(c)

# load pre-calculated fits
data = dict()
with open(join(dirname(__file__), 'test_data/data.pickle'), 'rb') as handle:
    data = pickle.load(handle)
melt_df = pd.DataFrame(data={'v50' : data['v50'], 'v50A' : data['v50A'], 'v50B' : data['v50B'],
                    'aicA' : data['aicA'], 'aicB' : data['aicB']})
source = ColumnDataSource(data=data)

# selection widgets
radio_group = RadioButtonGroup(
        labels=["Monophasic Tm", "Biphasic Tm-1", "Biphasic Tm-2"], width=400, active=0)
radio_group.on_change('active', radio_update)

checkbox_group = CheckboxGroup(
        labels=["Monophasic", "Biphasic"], active=[0])
checkbox_group.on_change('active', checkbox_update)

# heatmap
heatmap_hover = HoverTool(tooltips="""
        <div>
                <div>
                        <span style="font-size: 14px; font-weight: bold; ">@sample</span>
                        <span style="font-size: 14px; font-weight: bold; ">@v50</span>
                </div>
        </div>
        """
)

heatmap_df = pd.DataFrame(data={'xname' : source.data['xname'], 'yname' : source.data['yname'],
                'colors' : source.data['colors'], 'alphas' : source.data['alphas'], 'v50' : source.data['v50'],
                'sample' : source.data['sample']})
heatmap_source = ColumnDataSource(heatmap_df)
heatmap = figure(title="Plate View",
           x_axis_location="above", tools=["save, tap, reset", heatmap_hover],
           x_range=columns, y_range=rows)

heatmap.plot_width = 500
heatmap.plot_height = 400
heatmap.grid.grid_line_color = None
heatmap.axis.axis_line_color = None
heatmap.axis.major_tick_line_color = None
heatmap.axis.major_label_text_font_size = "12pt"
heatmap.axis.major_label_standoff = 0
heatmap.xaxis.major_label_orientation = np.pi/3

heatmap.rect('xname', 'yname', 0.9, 0.9, source=heatmap_source,
       color='colors', alpha='alphas', line_color=None,
       hover_line_color='black', hover_color='colors')
heatmap_row = TextInput(value = '', title = "Row index:")
heatmap_cell_column_1 = TextInput(value = '', title = "Sample:")

heatmap_source.on_change('selected', heatmap_select)

# text input
top_input = TextInput(value=" ", title="monophasic fit: top")
topa_input = TextInput(value=" ", title="biphasic fit: top-1")
topb_input = TextInput(value=" ", title="biphasic fit: top-2")
bottom_input = TextInput(value=" ", title="monophasic fit: bottom")
bottoma_input = TextInput(value=" ", title="biphasic fit: bottom-1")
bottomb_input = TextInput(value=" ", title="biphasic fit: bottom-2")
v50_input = TextInput(value=" ", title="monophasic fit: Tm")
v50a_input = TextInput(value=" ", title="biphasic fit: Tm-1")
v50b_input = TextInput(value=" ", title="biphasic fit: Tm-2")
slope_input = TextInput(value=" ", title="monophasic fit: slope")
slopeb_input = TextInput(value=" ", title="biphasic fit: slope-2")
slopea_input = TextInput(value=" ", title="biphasic fit: slope-1")

# datatable
table_dict = {'Sample' : source.data['sample'], 'Tm' : source.data['v50'], 'AIC monophasic' : source.data['aicA'],
                'probability' : source.data['probability']}
source_data_table = ColumnDataSource(table_dict)
columns = [
        TableColumn(field="Sample", title="Sample"),
        TableColumn(field="Tm", title="Tm ("+degree_sign+"C)"),
        TableColumn(field="Tm-1", title="Tm-1 ("+degree_sign+"C)"),
        TableColumn(field="Tm-2", title="Tm-2 ("+degree_sign+"C)"),
        TableColumn(field="AIC monophasic", title="AIC monophasic"),
        TableColumn(field="AIC biphasic", title="AIC biphasic"),
        TableColumn(field="probability", title="Max probability monophasic")
    ]
data_table = DataTable(source=source_data_table, columns=columns, width=500, height=400, selectable=True, editable=True)
table_row = TextInput(value = '', title = "Row index:")
table_cell_column_1 = TextInput(value = '', title = "Sample:")

source_data_table.on_change('selected', table_select)

# plot sample fit
sample_df = db[sample]
max_temp = sample_df[sample_df.y == np.max(sample_df.y)].x.values[0]
tmp_df = sample_df[sample_df.x <= max_temp]
min_temp = sample_df[sample_df.y == np.min(sample_df.y)].x.values[0]
tmp_df = tmp_df[tmp_df.x >= min_temp]
fit = model_fit(tmp_df, 'mono')
fitdf = pd.DataFrame(data={'x' : tmp_df.x, 'y' : fit.best_fit})
source_model = ColumnDataSource(data=dict(x=[], y=[]))
source_model_line_m = ColumnDataSource(data=dict(x=[], y=[]))
source_model_line_b = ColumnDataSource(data=dict(x=[], y=[]))
tools_model = 'xbox_select,wheel_zoom,pan,reset,save'
sample_ix = list(df)[1:].index(sample)
model = figure(title=sample+', Tm = '+str(source.data['v50A'][sample_ix]), x_axis_label="Temperature "+degree_sign+"C", y_axis_label="RFU",
               plot_width=600, plot_height=600, tools=tools_model)

source_model.data = source_model.from_df(sample_df)
source_model_line_m.data = source_model_line_m.from_df(fitdf[['x', 'y']])

model.circle('x', 'y', size=8, source=source_model, color='grey', alpha=0.6, legend='raw data')
model.line('x', 'y', source=source_model_line_m, line_width=3, color='red', alpha=0.4, legend='monophasic fit')
model.line('x', 'y', source=source_model_line_b, line_width=3, color='blue', alpha=0.4, legend='biphasic fit')

source_model.on_change('selected', selection_range)

# buttons
state = 0
file_source = ColumnDataSource({'file_contents':[], 'file_name':[]})
file_source.on_change('data', file_callback)
upload_button = Button(label="Upload Local File", button_type="success")
upload_button.callback = CustomJS(args=dict(file_source=file_source),
                            code=open(join(dirname(__file__), "upload.js")).read())

download_button = Button(label="Download Table to CSV", button_type="success")
output_filename = 'output.csv'
download_button.callback = CustomJS(args=dict(source=source_data_table, file_name=output_filename, state=state),
                            code=open(join(dirname(__file__), "download.js")).read())
copy_button = Button(label="Copy Tm's to Clipboard", button_type="success")

copy_button.callback = CustomJS(args=dict(source=source_data_table, state=state),
                        code=open(join(dirname(__file__), "copy.js")).read())

# page layout
desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=1000)

sizing_mode = 'fixed'
button_row = row(upload_button, download_button, copy_button)
selection_row = row(radio_group, checkbox_group)
init_inputa = widgetbox(bottom_input, top_input, v50_input, slope_input,
                    bottoma_input, topa_input)
init_inputb = widgetbox(v50a_input, slopea_input,
                    bottomb_input, topb_input, v50b_input, slopeb_input)
init_input = row(init_inputa, init_inputb)
main_row = row(column(heatmap, data_table), column(model, init_input))
l = layout([
    [desc],
    [button_row],
    [selection_row],
    [main_row],
], sizing_mode=sizing_mode)

update()


curdoc().add_root(l)
curdoc().title = 'Thermofluor'
