from dash import Dash, html, dcc, Input, Output, State
import altair as alt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_circles, make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

# Setup app and layout/frontend
app = Dash(__name__,  external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.layout = html.Div([
    html.Div([
        html.H2(
            html.A(
                "k-Means Simulation",
                href="https://github.com/stevenlio88/kmeans_simulation",
                style={
                    "text-decoration": "none",
                    "color": "inherit"
                        }
                    )
        ),
            html.Br(),
            html.Label("Number of points"),
            dcc.Input(
                id='N-widget',
                placeholder='Enter a value...',
                type='number',
                min=10, max=2500, step=1,
                value=500),
        
            #html.Br(),
            html.Label("Spread of the points"),        
            dcc.Slider(
                id='spread-widget',
                min=0,
                max=2,
                marks=None,
                value=0.25),
        
            #html.Br(),
            html.Label("Number of True Classes"),
            dcc.Slider(
                id='k-widget',
                min=1,
                max=10,
                step=1,               
                value=3),
        
            #html.Br(),
            html.Label("Number of k-Mean"),
            dcc.Slider(
                id='k2-widget',
                min=1,
                max=10,
                step=1,               
                value=4),
        
            #html.Br(),
            html.Label("Random Seed"),    
            dcc.Input(
                id='seed-widget',
                placeholder='Enter a value...',
                type='number',
                min=1, max=3000, step=1,
                value=150),
        ],style={'padding':5,'flex':0.5}
    ),

    html.Div([
        dcc.RadioItems(['Original', 'Predictions'], 
                        value='Predictions',
                        inline=True,
                        id='plotmode-widget'
                ),
        html.Iframe(
                    id='plot',
                    style={'border-width': '0', 'width': '100%', 'height': '400px'}
                ),
            ], style={'padding': 0, 'flex': 5}
        )
    ], style={'display': 'flex', 'flex-direction': 'row'}
)


# Set up callbacks/backend
@app.callback(
    Output('plot', 'srcDoc'),
    Input('N-widget', 'value'),
    Input('k-widget', 'value'),
    Input('k2-widget', 'value'),
    Input('spread-widget', 'value'),
    Input('seed-widget', 'value'),
    Input('plotmode-widget', 'value'))

def plot_kmean(N, k, k2, spread, seed, plotmode):
    if N is None:
        N=2500

    np.random.seed(seed)
    cstd=np.random.random_sample((k,))+spread

    X, y = make_blobs(n_samples=N, centers=k, center_box=(-5, 5), cluster_std=cstd, random_state=seed)

    k_means = KMeans(init="k-means++", n_clusters=k2, n_init=min(10, N))
    k_means.fit(X)
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels = k_means.labels_+1


    df = pd.DataFrame({'X':X[:,0], 'Y':X[:,1],'true class': y+1, 'pred class': k_means_labels})
    kdf = pd.DataFrame({'X':k_means_cluster_centers[:,0], 'Y':k_means_cluster_centers[:,1],'pred class': [i+1 for i in range(k2)]})

    if plotmode=="Original":
        chart2 = alt.Chart(df, title="Simulated Data").mark_circle(size=60).encode(
            x='X',
            y='Y',
            color=alt.Color('true class:N'),
            tooltip=['X','Y','true class','pred class'])
    else:
        chart = alt.Chart(df, title=f"k-Means Predictions with inertia: {k_means.inertia_:.2f}").mark_circle(size=60).encode(
            x='X',
            y='Y',
            color=alt.Color('pred class:N'),
            tooltip=['X','Y','true class','pred class'])

        centroid = alt.Chart(kdf).mark_point(size=80, filled=True, color='black').encode(
            x='X',
            y='Y',
            shape=alt.Shape('pred class:N', legend=None)
        )
                
        chart2 = (chart+centroid).interactive()

    return chart2.to_html()



if __name__ == '__main__':
    app.run_server(debug=True)
