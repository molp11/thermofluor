from os.path import join, dirname
from io import StringIO
import re
import base64
import pickle
import pandas as pd
import numpy as np
from lmfit import Model, Parameters
from bokeh.io import curdoc
from bokeh.layouts import row, column, layout
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, Div, BasicTickFormatter, Range1d, ColorBar, TapTool
from bokeh.models.widgets import Select, TextInput, Button, DataTable, TableColumn, RangeSlider, Slider, HTMLTemplateFormatter, CheckboxButtonGroup, RadioButtonGroup
from bokeh.plotting import figure
from bokeh.transform import linear_cmap
from bokeh.palettes import RdBu
from bokeh.events import Tap

degree_sign= u'\N{DEGREE SIGN}'

def make_well_ids(n_rows, n_cols, orientation):
    '''
    Make well IDs for plate object depending on whether data file is organized by rows or columns
    '''
    names = []
    
    if orientation == 0:
        for r in range(n_rows):
            for n in range(n_cols):
                names.append((r, n))
    else:
        for n in range(n_cols):
            for r in range(n_rows):
                names.append((r, n))
    
    return names


def make_param_dict():
    '''
    Make parameter dictionary based on user selections for passing to plate object
    '''
    param_dict['bottom'] = bottom_set_text.value
    param_dict['top'] = top_set_text.value
    param_dict['slope'] = slope_set_text.value
    
    if len(fix_bottom_checkbox.active) > 0:
        param_dict['bottom_vary'] = False
    else:
        param_dict['bottom_vary'] = True
    
    if len(fix_top_checkbox.active) > 0:
        param_dict['top_vary'] = False
    else:
        param_dict['top_vary'] = True
    
    if len(fix_slope_checkbox.active) > 0:
        param_dict['slope_vary'] = False
    else:
        param_dict['slope_vary'] = True
    
    return param_dict


def make_plate_source():
    '''
    Re-make dataframe for setting plate source
    '''
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    w, n, t, e = [], [], [], []
    
    if plate_type_buttons.active == 1:
        rows = 16
        columns = 24
    else:
        rows = 8
        columns = 12
        
    for i in range(rows):
        for j in range(columns):
            w.append(letters[i]+str(j+1))
            try:
                n.append(plate[(i, j)].name)
                t.append(plate[(i, j)].v50_fit)
                e.append(plate[(i, j)].v50_err)
            except:
                n.append('')
                t.append(np.nan)
                e.append(np.nan)
            
    xname = [x[1:] for x in w]
    yname = [y[0] for y in w]
    
    return pd.DataFrame(dict(w=w, n=n, t=t, e=e, 
                            xname=xname, yname=yname)).to_dict('list')
    
    
def fit_sample(name, param_dict):
    '''
    Reset ts_sample object in plate according to user selections
    '''
    if name in list(df):
        start = range_slider.value[0]
        stop = range_slider.value[1]

        fit_df = df[[df.columns[0], name]]
        fit_df = fit_df[(fit_df[fit_df.columns[0]] >= start) &
                    (fit_df[fit_df.columns[0]] <= stop)]

        fit_df.columns = ['x', 'y']

        return ts_sample(name, fit_df).boltzman_sigmoidal_model(param_dict)
    else:
        
        return None


def fit_new(name, param_dict):
        '''
        Fit new sample with no pre-defined selection range
        '''
        if name in list(df):
            fit_df = df[[df.columns[0], name]]
            fit_df.columns = ['x', 'y']
            midpoint = int(np.floor(len(fit_df)/2))
            idmin = fit_df.y[:midpoint].idxmin()
            idmax = fit_df.y[idmin:].idxmax()
            if (idmax - idmin) > 3: 
                fit_df = fit_df.iloc[idmin:idmax]
            
                return ts_sample(name, fit_df).boltzman_sigmoidal_model(param_dict)
            else:
                
                return ts_sample(name, fit_df)
        else:
            
            return None
            

class ts_sample(object):
    '''
    Contains curve fitting information for a given sample
    '''
    def __init__(self, name, dataframe):
        
        self.name = name
        self.data = dataframe
    
    def boltzman_sigmoidal_model(self, param_dict):
        '''
        calculate boltzman sigmoidal parameters using lmfit
        '''
        def boltzman_sigmoidal(x, bottom, top, v50, slope):
            '''
            Boltzman Sigmoidal equation
            '''
            return bottom + (top - bottom) / (1 + np.exp((v50 - x) / slope))
        
        self.bottom_vary = param_dict['bottom_vary']
        try:
            self.bottom = float(param_dict['bottom'])
        except:
            self.bottom = np.min(self.data.y)
            
        self.top_vary = param_dict['top_vary']
        try:
            self.top = float(param_dict['top'])
        except:
            self.top = np.max(self.data.y)
            
        self.v50_vary = True
        self.v50 = np.mean([np.max(self.data.x), np.min(self.data.x)])
        
        self.slope_vary = param_dict['slope_vary']
        try:
            self.slope = float(param_dict['slope'])
        except:
            self.slope = 1
        
        model = Model(boltzman_sigmoidal)
        
        params = Parameters()
        params.add(name='bottom', value=self.bottom, vary=self.bottom_vary)
        params.add(name='top', value=self.top, vary=self.top_vary)
        params.add(name='v50', value=self.v50, vary=self.v50_vary)
        params.add(name='slope', value=self.slope, vary=self.slope_vary)
        try:
            result = model.fit(self.data.y, params, x=self.data.x)
            fit_error = np.sqrt(np.diag(result.covar))

            self.y_fit = result.best_fit
            self.residuals = self.data.y - self.y_fit
            self.bottom_fit = np.round(result.params['bottom'].value, 2)
            self.bottom_err = np.round(result.params['bottom'].stderr, 2)
            self.top_fit = np.round(result.params['top'].value, 2)
            self.top_err = np.round(result.params['top'].stderr, 2)
            self.v50_fit = np.round(result.params['v50'].value, 2)
            self.v50_err = np.round(result.params['v50'].stderr, 2)
            self.slope_fit = np.round(result.params['slope'].value, 2)
            self.slope_err = np.round(result.params['slope'].stderr, 2)
        except:
            self.y_fit = np.empty(len(self.data['y'].values))
            self.residuals = np.empty(len(self.data['y'].values))
            self.bottom_fit = np.nan
            self.bottom_err = np.nan
            self.top_fit = np.nan
            self.top_err = np.nan
            self.v50_fit = np.nan
            self.v50_err = np.nan
            self.slope_fit = np.nan
            self.slope_err = np.nan
            
        return self
    

def parameter_set_callback(attrname, old, new):
    '''
    Refit current sample based on parameter selections
    '''
    param_dict = make_param_dict()
    plate[well_id] = fit_sample(plate[well_id].name, param_dict)
    
    update()
    
    
def slider_callback(attrname, old, new):
    '''
    Refit current sample based on slider values
    '''
    start = float(range_slider.value[0])
    stop = float(range_slider.value[1])
    step = float(range_slider.step)
    end = float(range_slider.end)
    
    if start >= stop -  5*step and stop <= end - 5*step:
        range_slider.value = (start, start + 5*step)
    
    elif start >= stop - 5*step and stop > end - 5*step:
        range_slider.value = (stop - 5*step, stop)
    
    param_dict = make_param_dict()
    global well_id
    global plate
    plate[well_id] = fit_sample(plate[well_id].name, param_dict)
    
    update()
    
    
def table_select(attrname, old, new):
    '''
    Update sample based on table selection
    '''
    index = plate_source.selected.indices[-1]
    global well_id
    if plate_type_buttons.active == 1:
        well_id = (np.floor_divide(index, 24), np.remainder(index, 24))
    else:
        well_id = (np.floor_divide(index, 12), np.remainder(index, 12))
    global sample
    sample = plate[well_id]
    range_slider.value = (sample.data['x'].values[0], sample.data['x'].values[-1])
    

def plate_select(event):
    '''
    Update sample based on plate map selection
    '''
    index = plate_source.selected.indices[-1]
    global well_id
    if plate_type_buttons.active == 1:
        well_id = (np.floor_divide(index, 24), np.remainder(index, 24))
    else:
        well_id = (np.floor_divide(index, 12), np.remainder(index, 12))
    global sample
    sample = plate[well_id]
    range_slider.value = (sample.data['x'].values[0], sample.data['x'].values[-1])


def plate_type_callback(attrname, old, new):
    '''
    Reload page with plate type set to 96 or 384 wells
    '''
    global plate_type
    plate_type = plate_type_buttons.active
    if plate_type == 1:
        rows = 16
        columns = 24
    else:
        rows = 8
        columns = 12
    
    global well_ids
    well_ids = make_well_ids(rows, columns, plate_layout)
        
    names = []
    count = 0
    global plate
    for w in well_ids:
        if w in list(plate):
            names.append(list(df)[count + 1])
            count+=1
        else:
            names.append('')
    
    param_dict = make_param_dict()
    new_plate = {}
    for name, wid in zip(names, well_ids):
        new_plate[wid] = fit_new(name, param_dict)

    plate = new_plate
    
    load_page(plate)
    

def plate_layout_callback(attrname, old, new):
    '''
    Update plate map based on data file layout selected by user
    '''
    global plate_layout
    plate_layout = plate_layout_buttons.active
    global plate_type
    plate_type = plate_type_buttons.active
    if plate_type == 1:
        rows = 16
        columns = 24
    else:
        rows = 8
        columns =12
        
    new_well_ids = make_well_ids(rows, columns, plate_layout)
    global plate
    global well_ids
    
    new_plate = {}
    for wid, nwid in zip(well_ids, new_well_ids):
        new_plate[nwid] = plate[wid]
    
    plate = new_plate
    well_ids = new_well_ids
    plate_source.data = make_plate_source()
    
    update()
    

def parameter_set_callback(attrname, old, new):
    '''
    Refit current sample based on parameter selections
    '''
    global well_id
    global plate
    param_dict = make_param_dict()
    plate[well_id] = fit_sample(plate[well_id].name, param_dict)
    
    update()
    
    
def refit_all_callback(event):
    '''
    Refit all samples with current user selections
    '''
    param_dict = make_param_dict()

    global plate
    plate = {}
    for name, wid in zip(df.columns[1:], well_ids):
        plate[wid] = fit_new(name, param_dict)
        
    plate_source.data = make_plate_source()
    
    update()
    
    
def update():
    '''
    Bokeh update method
    '''
    global well_id
    global plate
    sample = plate[well_id]
    sample_source.data = pd.DataFrame(data=dict(x=sample.data.x, y=sample.data.y, 
                                fit=sample.y_fit, residuals=sample.residuals)).to_dict('list')
    sample_scatter.title.text = sample.name + ' fit'
    
    if plate_type_buttons.active == 1:
        ix = well_id[0]*24 + well_id[1]
    else:
        ix = well_id[0]*12 + well_id[1]
    w = plate_source.data['w']
    n = plate_source.data['n']
    t = plate_source.data['t']
    e = plate_source.data['e']    
    t[ix] = sample.v50_fit
    e[ix] = sample.v50_err
    xname = [x[1:] for x in w]
    yname = [y[0] for y in w]
    
    mapper = linear_cmap(field_name='t', palette=RdBu[8], low=min(t), high=max(t))
    color_bar.color_mapper = mapper['transform']
    
    plate_source.data = pd.DataFrame(dict(w=w, n=n, t=t, e=e, 
                                          xname=xname, yname=yname)).to_dict('list')
    
    sample_table_source.data = pd.DataFrame(dict(l = ['Fit value', 'Std. error'],
                                                 b = [sample.bottom_fit, sample.bottom_err],
                                                 t = [sample.top_fit, sample.top_err],
                                                 v = [sample.v50_fit, sample.v50_err],
                                                 s = [sample.slope_fit, sample.slope_err]))
    

def file_callback(attrname, old, new):
    '''
    Read uploaded data and use to load new page
    '''
    raw_contents = file_source.data['file_contents'][0]
    prefix, b64_contents = raw_contents.split(',', 1)
    file_contents = base64.b64decode(b64_contents).decode('utf-8-sig', errors='ignore')
    file_io = StringIO(file_contents)
    
    global df
    df = pd.read_csv(file_io)
    df.columns = [str(s) for s in list(df)]
    
    global plate
    plate = {}
    
    global well_ids
    if len(list(df)) > 97:
        well_ids = make_well_ids(16, 24, plate_layout)
    else:
        well_ids = make_well_ids(8, 12, plate_layout)
    param_dict = {'bottom' : None, 'bottom_vary' : True,
                  'top' : None, 'top_vary' : True,
                  'slope' : None, 'slope_vary' : True
                 }

    for name, wid in zip(df.columns[1:], well_ids):
        fit_df = df[[df.columns[0], name]]
        fit_df.columns = ['x', 'y']
        try:
            idmin = fit_df.y.idxmin()
            idmax = fit_df.y[idmin:].idxmax()
            if (idmax - idmin) > 3: 
                fit_df = fit_df.iloc[idmin:idmax]
                plate[wid] = ts_sample(name, fit_df).boltzman_sigmoidal_model(param_dict)
            else:
                plate[wid] = ts_sample(name, fit_df).boltzman_sigmoidal_model(param_dict)
        except:
            plate[wid] = ts_sample(name, fit_df).boltzman_sigmoidal_model(param_dict)
            
    load_page(plate)
    
    
def load_page(plate):
    '''
    Load new page
    '''
    global well_id 
    well_id = (0, 0)
    
    global sample 
    sample = plate[well_id]
    
    # Button to upload local file
    global file_source
    file_source = ColumnDataSource(data=dict(file_contents = [], file_name = []))
    file_source.on_change('data', file_callback)
    try:
        output_file_name = file_source.data['file_name'] + '-out.csv'
    except:
        output_filename = 'output.csv'
    global upload_button
    upload_button = Button(label="Upload local file", button_type="success", width=200, height=30)
    upload_button.js_on_click(CustomJS(args=dict(file_source=file_source),
                               code=open(join(dirname(__file__), "upload.js")).read()))
    
    # Text boxes for setting fit parameters
    global bottom_set_text
    bottom_set_text = TextInput(value='', title="Set initial value for Fmin", width=200, height=50)
    bottom_set_text.on_change('value', parameter_set_callback)

    global top_set_text
    top_set_text = TextInput(value='', title="Set initial value for Fmax", width=200, height=50)
    top_set_text.on_change('value', parameter_set_callback)
    
    global slope_set_text
    slope_set_text = TextInput(value='', title="Set initial value for a", width=200, height=50)
    slope_set_text.on_change('value', parameter_set_callback)
    
    # Radio button group for setting plate type
    global plate_type_buttons
    global plate_type
    plate_type_buttons = RadioButtonGroup(labels=['96 well', '384 well'], 
                                          width=200, height=25, active=plate_type)
    plate_type_buttons.on_change('active', plate_type_callback)
    
    # Radio button group for setting data layout
    global plate_layout_buttons
    global plate_layout
    plate_layout_buttons = RadioButtonGroup(labels=['by row', 'by column'],
                                           width=200, height=25, active=plate_layout)
    plate_layout_buttons.on_change('active', plate_layout_callback)
    
    # Checkbox groups for fixing fit parameters
    global fix_bottom_checkbox
    fix_bottom_checkbox = CheckboxButtonGroup(labels=['Fix min fluoresence (Fmin)'], 
                                              width=200, height=30)
    fix_bottom_checkbox.on_change('active', parameter_set_callback)
    
    global fix_top_checkbox
    fix_top_checkbox = CheckboxButtonGroup(labels=['Fix max fluorescence (Fmax)'], width=200, height=30)
    fix_top_checkbox.on_change('active', parameter_set_callback)
    
    global fix_slope_checkbox
    fix_slope_checkbox = CheckboxButtonGroup(labels=['Fix curve shape parameter (a)'], width=200, height=30)
    fix_slope_checkbox.on_change('active', parameter_set_callback)
    
    # Slider for selecting data to fit
    global df
    xmin = df[df.columns[0]].values[0]
    xstep = df[df.columns[0]].values[1] - xmin
    xstart = sample.data['x'].values[0]
    xend = sample.data['x'].values[-1]
    xmax = df[df.columns[0]].values[-1]
    
    global range_slider
    range_slider = RangeSlider(start=xmin, end=xmax, value=(xstart, xend),
                    step=xstep,
                    title='Fine tune temperature range', width=550)
    range_slider.on_change('value', slider_callback)
    
    # Scatter plot for fitting individual samples
    global sample_source
    sample_source = ColumnDataSource(data=dict(x=sample.data.x, y=sample.data.y, 
                                        fit=sample.y_fit, residuals=sample.residuals))
    global sample_scatter
    plot_tools = 'wheel_zoom, pan, reset, save'
    sample_scatter = figure(title="Boltzman sigmoidal fit", x_axis_label='Temperature ('+degree_sign+'C)',
                            y_axis_label="Fluoresence intensity", plot_width=600, 
                            plot_height=300, tools=plot_tools)
    sample_scatter.circle(x='x', y='y', color='grey', size=8, alpha=0.6, source=sample_source)
    sample_scatter.line(x='x', y='fit', color='black', line_width=2, 
                        alpha=1.0, source=sample_source)
    sample_scatter.title.text = sample.name + ' fit'
    
    # Scatter plot for residuals of individual sample fit
    global residual_scatter
    residual_scatter = figure(title="Fit residuals", x_axis_label='Temperature ('+degree_sign+'C)',
                              y_axis_label="Residual", plot_width=600, 
                              plot_height=200, tools='wheel_zoom,pan,reset')
    residual_scatter.yaxis.formatter = BasicTickFormatter(precision=2, use_scientific=True)
    residual_scatter.circle('x', 'residuals', size=8, source=sample_source, 
                            color='grey', alpha=0.6)
    
    # Heatmap for displaying all Tm values in dataset
    global plate_source
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    w, n, t, e = [], [], [], []
    if plate_type_buttons.active == 1:
        rows = 16
        columns = 24
    else:
        rows = 8
        columns = 12
    
    for i in range(rows):
        for j in range(columns):
            w.append(letters[i]+str(j+1))
            try:
                n.append(plate[(i, j)].name)
                t.append(plate[(i, j)].v50_fit)
                e.append(plate[(i, j)].v50_err)
            except:
                n.append('')
                t.append(np.nan)
                e.append(np.nan)
                
    xname = [x[1:] for x in w]
    yname = [y[0] for y in w]
    
    plate_source = ColumnDataSource(dict(w=w, n=n, t=t, e=e, xname=xname, yname=yname)) 
    plate_columns = [
        TableColumn(field='w', title='Well ID'),
        TableColumn(field='n', title='Sample name'),
        TableColumn(field='t', title='Tm ('+degree_sign+'C)'),
        TableColumn(field='e', title='Error ('+degree_sign+'C)'),
    ]
    
    plate_map_hover = HoverTool(tooltips="""
        <div>
                <div>
                        <span style="font-size: 14px; font-weight: bold; ">@n:</span>
                        <span style="font-size: 14px; font-weight: bold; ">@t</span>
                </div>
        </div>
        """
    )
    
    if plate_type_buttons.active == 1:
        plate_map = figure(title="Plate view", x_axis_location="above", height=400, width=620, 
                       tools=["save, tap, reset", plate_map_hover], 
                       x_range=[str(x+1) for x in range(0, columns)]+['', 'Tm ('+degree_sign+'C)'],
                       y_range=letters[:rows][::-1])
    else:
        plate_map = figure(title="Plate view", x_axis_location="above", height=400, width=620, 
                       tools=["save, tap, reset", plate_map_hover], 
                       x_range=[str(x+1) for x in range(0, columns)]+['Tm ('+degree_sign+'C)'], 
                       y_range=letters[:rows][::-1])
        
    taptool = plate_map.select(type=TapTool)
    plate_map.on_event(Tap, plate_select)
    
    global mapper
    mapper = linear_cmap(field_name='t', palette=RdBu[8], low=min(t), high=max(t))
    
    global color_bar
    color_bar = ColorBar(color_mapper=mapper['transform'], width=10, height=250, name='Tm ('+degree_sign+'C)')
    plate_map.add_layout(color_bar, 'right')
    
    plate_map.grid.grid_line_color = None
    plate_map.axis.axis_line_color = None
    plate_map.axis.major_tick_line_color = None
    plate_map.axis.major_label_text_font_size = "10pt"
    plate_map.axis.major_label_standoff = 0
    plate_map.rect('xname', 'yname', .95, .95, source=plate_source,
            color=mapper, line_color='black', line_width=1)
    
    # Table listing all Tm values in dataset
    global plate_table
    plate_table = DataTable(source=plate_source, columns=plate_columns, width=500,
                            height=500, selectable=True, editable=True)
    plate_table.source.selected.on_change('indices', table_select)
    
    # Table showing fitting parameters for current sample
    global sample_table_source
    sample_table_source = ColumnDataSource(data=dict(l=['Fit value', 'Std. error'],
                                                     b=[sample.bottom_fit, sample.bottom_err],
                                                     t=[sample.top_fit, sample.top_err],
                                                     v=[sample.v50_fit, sample.v50_err],
                                                     s=[sample.slope_fit, sample.slope_err])) 
    sample_table_columns = [
        TableColumn(field='l', title=''),
        TableColumn(field='b', title='Fmin'),
        TableColumn(field='t', title='Fmax'),
        TableColumn(field='v', title='Tm ('+degree_sign+'C)'),
        TableColumn(field='s', title='a')
    ]
    global sample_table
    sample_table = DataTable(source=sample_table_source, columns=sample_table_columns, width=600,
                            height=200, selectable=False, editable=False)
   
    # Button to re-fit all with current parameter settings
    global refit_all_button
    refit_all_button = Button(label="Re-fit all samples", 
                              button_type='danger', width=200, height=30)
    refit_all_button.on_click(refit_all_callback)
    
    # Button to download Tm table to csv file
    global download_button
    download_button = Button(label="Download table to CSV", 
                             button_type="primary", width=200, height=30)
    download_button.js_on_click(CustomJS(args=dict(source=plate_source, file_name=output_filename), 
                                        code=open(join(dirname(__file__), "download.js")).read()))

    # Button to copy Tm table to clipboard
    global copy_button
    copy_button = Button(label="Copy table to clipboard", button_type="primary", 
                         width=200, height=30)
    copy_button.js_on_click(CustomJS(args=dict(source=plate_source),
                               code=open(join(dirname(__file__), "copy.js")).read()))

    # page formatting
    desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=1200)
    main_row = row(column(plate_type_buttons, plate_layout_buttons, upload_button, 
                          fix_bottom_checkbox, bottom_set_text, fix_top_checkbox, 
                          top_set_text, fix_slope_checkbox, slope_set_text, refit_all_button,
                          download_button, copy_button),
                   column(sample_scatter, residual_scatter, range_slider, sample_table),
                   column(plate_map, plate_table))
        
    sizing_mode = 'scale_width'
    l = layout([
        [desc],
        [main_row]
    ], sizing_mode=sizing_mode)
    
    update()
    curdoc().clear()
    curdoc().add_root(l)
    curdoc().title = "DSF"
    
    
data_file = join(dirname(__file__), 'test_data/test.csv')
df = pd.read_csv(data_file)
df.columns = [str(s) for s in list(df)]

plate = {}
plate_type = 0
plate_layout = 0
well_ids = make_well_ids(8, 12, plate_layout)
param_dict = {'bottom' : None, 'bottom_vary' : True,
              'top' : None, 'top_vary' : True,
              'slope' : None, 'slope_vary' : True
             }

filehandler = open(join(dirname(__file__), 'test_data/' + 'data_dict.pkl'), 'rb')
data_dict = pickle.load(filehandler)

for ix, wid in enumerate(well_ids):
    sample = ts_sample(data_dict['name'][ix], pd.DataFrame(columns=['x', 'y']))
    sample.data['x'] = data_dict['x'][ix]
    sample.data['y'] = data_dict['y'][ix]
    sample.bottom_vary = data_dict['bottom_vary'][ix]
    sample.bottom = data_dict['bottom'][ix]
    sample.top_vary = data_dict['top_vary'][ix]
    sample.top = data_dict['top'][ix]
    sample.v50_vary = data_dict['v50_vary'][ix]
    sample.v50 = data_dict['v50'][ix]
    sample.slope_vary = data_dict['slope_vary'][ix]
    sample.slope = data_dict['slope'][ix]
    sample.y_fit = data_dict['y_fit'][ix]
    sample.residuals = data_dict['residuals'][ix]
    sample.bottom_fit = data_dict['bottom_fit'][ix]
    sample.bottom_err = data_dict['bottom_err'][ix]
    sample.top_fit = data_dict['top_fit'][ix]
    sample.top_err = data_dict['top_err'][ix]
    sample.v50_fit = data_dict['v50_fit'][ix]
    sample.v50_err = data_dict['v50_err'][ix]
    sample.slope_fit = data_dict['slope_fit'][ix]
    sample.slope_err = data_dict['slope_err'][ix]    
    plate[wid] = sample
   
load_page(plate)