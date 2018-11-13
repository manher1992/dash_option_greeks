# Import required libraries
import os
from random import randint

import plotly.plotly as py
from plotly.graph_objs import *

import flask
import dash
from dash.dependencies import Input, Output, State, Event
import dash_core_components as dcc
import dash_html_components as html


# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))
app = dash.Dash(__name__, server=server)


"""
Created on Sun Oct 28 13:12:10 2018

@author: Herdt
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
import mpl_toolkits.mplot3d.axes3d as p3

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'serif'
from scipy.integrate import quad

def dN(x):
    ''' Probability density function of standard normal random variable x.'''
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
def N(d):
    ''' Cumulative density function of standard normal random variable x. '''
    return quad(lambda x: dN(x), -20, d, limit=50)[0]
def d1f(St, K, t, T, r, sigma):
    ''' Black-Scholes-Merton d1 function.
        Parameters see e.g. BSM_call_value function. '''
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
            * (T - t)) / (sigma * math.sqrt(T - t))
    return d1




def BSM_delta(St, K, t, T, r, sigma):
    d1 = d1f(St, K, t, T, r, sigma)
    delta = N(d1)
    return delta

def BSM_gamma(St, K, t, T, r, sigma):
    d1 = d1f(St, K, t, T, r, sigma)
    gamma = dN(d1) / (St * sigma * math.sqrt(T - t))
    return gamma

def BSM_theta(St, K, t, T, r, sigma):
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    theta = -(St * dN(d1) * sigma / (2 * math.sqrt(T - t)) + r * K * math.exp(-r * (T - t)) * N(d2))
    return theta

def BSM_rho(St, K, t, T, r, sigma):
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    rho = K * (T - t) * math.exp(-r * (T - t)) * N(d2)
    return rho

def BSM_vega(St, K, t, T, r, sigma):
    d1 = d1f(St, K, t, T, r, sigma)
    vega = St * dN(d1) * math.sqrt(T - t)
    return vega

#def plot_greeks(function, greek):
#    # Model Parameters
#    St = 100.0 # index level
#    K = 100.0 # option strike
#    t = 0.0 # valuation date
#    T = 1.0 # maturity date
#    r = 0.05 # risk-less short rate
#    sigma = 0.2 # volatility
#    
#    
#    tlist = np.linspace(0.01, 1, 25)
#    klist = np.linspace(80, 120, 25)
#    V = np.zeros((len(tlist), len(klist)), dtype=np.float)
#    for j in range(len(klist)):
#        for i in range(len(tlist)):
#            V[i, j] = function(St, klist[j], t, tlist[i], r, sigma)
#            
#    x, y = np.meshgrid(klist, tlist)
#    fig = plt.figure(figsize=(9, 5))
#    plot = p3.Axes3D(fig)
#    plot.plot_wireframe(x, y, V)
#    plot.set_xlabel('strike $K$')
#    plot.set_ylabel('maturity $T$')
#    plot.set_zlabel('%s(K, T)' % greek)



import plotly.graph_objs as go
import pandas as pd




    

greek_options = ['Delta','Gamma','Vega','Theta','Rho']

#

app.layout = html.Div([
    html.H2("Greek"),
    html.Div(
        [
            dcc.Dropdown(
                id="greek_options",
                options=[{
                    'label': i,
                    'value': i
                } for i in greek_options],
                value=''),
        ],
        style={'width': '25%',
               'display': 'inline-block'}),
    dcc.Graph(id='greeks-graph'),
])


@app.callback(
    dash.dependencies.Output('greeks-graph', 'figure'),
    [dash.dependencies.Input('greek_options', 'value')])


def greeks(greek_options):
    if greek_options == 'Delta':        
        greek_options = BSM_delta
    elif greek_options == 'Gamma':
        greek_options = BSM_gamma
    elif greek_options == 'Theta':
        greek_options = BSM_theta
    elif greek_options == 'Vega':
        greek_options = BSM_vega

    elif greek_options == 'Rho':
        greek_options = BSM_rho    




    # Model Parameters
    St = 100.0 # index level
    K = 100.0 # option strike
    t = 0.0 # valuation date
    T = 1.0 # maturity date
    r = 0.05 # risk-less short rate
    sigma = 0.2 # volatility
    
    
    tlist = np.linspace(0.01, 1, 25)
    klist = np.linspace(80, 120, 25)
    V = np.zeros((len(tlist), len(klist)), dtype=np.float)
    for j in range(len(klist)):
        for i in range(len(tlist)):
            V[i, j] = greek_options(klist[j],K, t, tlist[i], r, sigma)
        
    x, y = np.meshgrid(klist, tlist)    
    #surface = go.Surface(x=x, y=y, z=V)
    #data = [surface]
    
    figure = {
            'data' : [
                    dict({
                            'x':x,
                            'y':y,
                            'z':V,
                            'mode':'markers',
                            'type':'surface'})]}

    return figure


# Run the Dash app
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
