import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import statsmodels.api as sm
import numpy as np
import dash_table
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import r2_score

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
app = dash.Dash(external_stylesheets = external_stylesheets)
app.title = "Autovero"
server = app.server

df = pd.read_csv("autovero.csv")

empty_layout = go.Layout(
	title = None,
	plot_bgcolor = "rgb(30, 30, 30)",
    paper_bgcolor = "rgb(30, 30, 30)")

app.layout = html.Div([
	html.H1("Autovero", style = {"textAlign":"center", "margin-bottom":"20", "margin-top": "20"}, className = "row"),

	html.Div([
		html.P("Merkki:", className = "four columns"),
		html.P("Malli:", className = "four columns", style = {"margin-left": 5}),
		html.P("Mallin tarkennin:", className = "four columns", style = {"margin-left": 10}),
		], className = "container"),

	html.Div([
		dcc.Dropdown(
			id = "Dropdown",
			options = [{"label": i, "value": i} for i in df.Merkki.value_counts().index.tolist()],
			className = "four columns",
			),
		dcc.Dropdown(
			id = "Dropdown2",
			className = "four columns",
			style = {"margin-left": 5}
			),
		dcc.Dropdown(
			id = "Dropdown3",
			className = "four columns",
			style = {"margin-left": 10}
			),
		], className = "container", style = {"color": "black"}),

	html.H2("multivariate regression", className = "row", style = {"textAlign": "center", "margin-bottom": 50, "margin-top" : 50}),

	html.Div([
		dcc.Graph(id = "3d_plot", className = "four columns"),
		dcc.Graph(id = "bar_fig", className = "four columns"),
		html.H5(id = "text", className = "four columns", style = {"padding-right": 10}),
		], className = "row"),

	html.H3("Hae veroennustetta antamalla ajetut kilometrit:", className = "row", style = {"textAlign": "center"}),
	html.Div([
		html.P("Ajokm/1000 (min):", className = "two columns"),
		html.P("Ajokm/1000 (max):", className = "two columns"),
		#html.P("Anna ika päivinä:", className = "two columns"),
		], className = "container", style = {"margin-left": 760}),

	html.Div([
		dcc.Input(id = "input", type = "text", className = "two columns"),
		dcc.Input(id = "input2", type = "text", className = "two columns"),
		#dcc.Input(id = "input3", type = "text", className = "two columns"),
		], className = "container", style = {"margin-left": 760, "margin-bottom": 25}),

	html.Div([ 
		html.H5(id = "text2", style = {"width": "auto"})
		], className = "container")

	], className = "row", style = {"color": "white", "padding-bottom": 200, "backgroundColor": "rgb(30, 30, 30)"})#, style = {"backgroundColor": "rgb(30, 30, 30)"

def mreg(merkki, malli, mallin_tarkennin, km1, km2):
	me = df[df["Merkki"] == merkki]
	ma = me[me["Malli"] == malli]
	mt = ma[ma["Mallin tarkennin"] == mallin_tarkennin]
	
	if len(mt) < 20:
		fig = go.Figure(data = [], layout = empty_layout)
		return "Datamäärä liian pieni", fig, fig, None
	else:
		X = mt.drop(["Merkki", "Malli", "Mallin tarkennin", "Autovero", "teho", "vaihteisto", "moottorikoko"],1)
		y = mt["Autovero"]

		X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		regressor = LinearRegression()
		regressor.fit(X_train, y_train)
		y_pred = regressor.predict(x_test)

		df2 = pd.DataFrame({"Actual": y_test, "Prediction": y_pred})
		df3 = df2.head(90)

		r2 = r2_score(y_test, y_pred)
		mae = metrics.mean_absolute_error(y_test, y_pred).round(2)
		mse = metrics.mean_squared_error(y_test, y_pred).round(2)
		rmsqe = np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(2)
		tulokset = pd.DataFrame(columns = ["tulos"], index = ["Keskimääräinen autovero", "Keskimääräinen auton ika", "Keskimääräinen Ajokm/1000", "r2_score", "Mean Absolute Erro", "Mean Squared Error",
															"Root Mean Squared Error", "Data size", "X_train size", "X_test size", "Y_train size", "Y_test size"],
			data = [y_pred.mean().round(2), round(mt.ika.mean()), round(mt["Ajokm/1000"].mean()), r2, mae, mse, rmsqe, len(mt), len(X_train), len(x_test), len(y_train), len(y_test)])

		data = mt.loc[df2.index]
		data.drop(["moottorikoko", "teho", "vaihteisto", "Autovero"],1, inplace=True)
		arvio_data = pd.concat([data, df2.Prediction.astype(int)], axis = 1)
		tulos = arvio_data[(arvio_data["Ajokm/1000"] >= km1) & (arvio_data["Ajokm/1000"] <= km2)]
		#tulos.drop(["Autovero", "moottorikoko", "teho", "vaihteisto"],1, inplace=True)

		if len(tulos) == 0:
			return datatable(tulokset.reset_index()), bar_plot(df3, mallin_tarkennin), kolmed_plot(mallin_tarkennin), html.P("Ei hakutuloksia", style = {"margin-left": 400})
		else:
			return datatable(tulokset.reset_index()), bar_plot(df3, mallin_tarkennin), kolmed_plot(mallin_tarkennin), datatable(tulos)

def datatable(dataframe):
	return dash_table.DataTable(
		data = dataframe.to_dict("records"),
		columns = [{"id": c, "name": c} for c in dataframe.columns],
	    style_header={"backgroundColor": "rgb(30, 30, 30)"},
	    #sorting = True,
	    style_cell={
	        "backgroundColor": "rgb(50, 50, 50)",
	        "color": "white",
	        "textAlign": "center",
        },
        style_table={
	        "maxHeight": "400px",
	        "overflowY": "scroll",
	        "maxWidth": "100%"
    	},
	)

def bar_plot(df, automalli):
	trace = go.Bar(
		name = "Prediction",
		y = df.Prediction
	)
	trace2 = go.Bar(
		name = "Actual",
		y = df.Actual
	)
	layout = go.Layout(
		title = automalli,
		font = dict(color = "white"),
		plot_bgcolor = "rgb(30, 30, 30)",
    	paper_bgcolor=  "rgb(30, 30, 30)",
    	legend = dict(orientation = "h")
	)
	data = [trace, trace2]
	fig = go.Figure(data = data, layout = layout)
	return fig

def kolmed_plot(mallin_tarkennin):
	df2 = df[df["Mallin tarkennin"] == mallin_tarkennin]

	X = df2[["ika", "Ajokm/1000"]]
	y = df2["Autovero"]
	X = sm.add_constant(X)
	est = sm.OLS(y, X).fit()
	xx1, xx2 = np.meshgrid(np.linspace(X.ika.min(), X.ika.max(), 100), 
	                       np.linspace(X["Ajokm/1000"].min(), X["Ajokm/1000"].max(), 100))

	Z = est.params[0] + est.params[1] * xx1 + est.params[2] * xx2

	surface = go.Surface(
	    x = xx1,
	    y = xx2,
	    z = Z,
	    showscale = False,
	    colorscale = "Greys",
	    opacity = 0.85
	)

	resid = y - est.predict(X)

	trace = go.Scatter3d(
	    x = X[resid >= 0].ika,
	    y = X[resid >= 0]["Ajokm/1000"],
	    z = y[resid >= 0],
	    mode = "markers",
	    marker = dict(size = 2, color = "white", opacity = 1, line = dict(color = "black", width = 0.2))
	)

	trace2 = go.Scatter3d(
	    x = X[resid < 0].ika,
	    y = X[resid < 0]["Ajokm/1000"],
	    z = y[resid < 0],
	    mode = "markers",
	    marker = dict(size = 2, color = "black", opacity = 1, line = dict(color = "white", width = 0.2))
	)

	layout = go.Layout(
    scene = dict(
        xaxis = dict(title = "Auton käyttöikä/pv",
            backgroundcolor = "rgb(30, 30, 30)",
            gridcolor = "rgb(255, 255, 255)",
            showbackground = True,
            zerolinecolor = "rgb(30, 30, 30)",
            ),
        
        yaxis = dict(title = "Ajokm/1000",
            backgroundcolor = "rgb(30, 30, 30)",
            gridcolor ="rgb(255, 255, 255)",
            showbackground = True,
            zerolinecolor = "rgb(30, 30, 30)",
        ),
        zaxis = dict(title = "Autovero",
            backgroundcolor = "rgb(30, 30, 30)",
            gridcolor = "rgb(255, 255, 255)",
            showbackground = True,
            zerolinecolor = "rgb(30, 30, 30)",
            
        ),
    ),
    title = "3D multivariate regression " + mallin_tarkennin,
    font = dict(color = "white"),
    autosize = True,
    #width = 600,
    #height = 550,
    showlegend = False,
    plot_bgcolor = "rgb(30, 30, 30)",
    paper_bgcolor = "rgb(30, 30, 30)",
	)

	data = [surface, trace, trace2]


	fig = dict(data = data, layout = layout)
	return fig

@app.callback(
    Output("Dropdown2", "options"),
	[Input("Dropdown", "value")])
def update_dropdown(selected):
	if selected is not None:
		merkki = df[df["Merkki"] == selected]["Malli"].value_counts().index.tolist()
		return [{"label": i, "value": i} for i in merkki]
	else:
		return []

@app.callback(
    Output("Dropdown3", "options"),
    [Input("Dropdown2", "value"),
    Input("Dropdown", "value")])
def update_dropdown2(selected, selected2):
	if selected and selected2 is not None:
		malli = df[df["Malli"] == selected]["Mallin tarkennin"].value_counts().index.tolist()
		return [{"label": i, "value": i} for i in malli]
	else:
		return []

@app.callback(
	[Output("text", "children"),
	Output("bar_fig", "figure"),
	Output("3d_plot", "figure"),
	Output("text2", "children")],
	[Input("Dropdown", "value"),
	Input("Dropdown2", "value"),
	Input("Dropdown3", "value"),
	Input("input", "n_submit"),
	Input("input", "n_blur"),
	Input("input2", "n_submit"),
	Input("input", "n_blur")],
	[State("input", "value"),
	State("input2", "value")])

def multi_reg(merkki, malli, mallin_tarkennin, ns1, nb1, ns2, nb2, km1, km2):
	km1 = pd.to_numeric(km1)
	km2 = pd.to_numeric(km2)
	if (merkki and malli and mallin_tarkennin is not None) and (km1 and km2 is ""):
		return mreg(merkki, malli, mallin_tarkennin, None, None)
	if merkki and malli and mallin_tarkennin and km1 and km2 is not None:
		return mreg(merkki, malli, mallin_tarkennin, km1, km2)
	else:
		fig = go.Figure(data = [], layout = empty_layout)
		return None, fig, fig, None

if __name__ == "__main__":
    app.run_server(debug = True, port = 7788)