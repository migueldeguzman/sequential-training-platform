import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

class PhaseTaggedPowerViz:
    def __init__(self, data_file):
        """
        Initialize visualization with phase-tagged power sample data
        """
        self.df = pd.read_csv(data_file)
        self.app = dash.Dash(__name__)
        self._create_layout()
    
    def _create_layout(self):
        """
        Create interactive dashboard layout
        """
        self.app.layout = html.Div([
            html.H1('Phase-Tagged Power Profiling'),
            dcc.Dropdown(
                id='phase-selector',
                options=[{'label': phase, 'value': phase} 
                         for phase in self.df['phase'].unique()],
                multi=True,
                placeholder='Select Phases'
            ),
            dcc.Graph(id='power-trace')
        ])
        
        @self.app.callback(
            Output('power-trace', 'figure'),
            Input('phase-selector', 'value')
        )
        def update_graph(selected_phases):
            """
            Update graph based on phase selection
            """
            if not selected_phases:
                return {}
            
            filtered_df = self.df[self.df['phase'].isin(selected_phases)]
            
            fig = px.line(
                filtered_df, 
                x='timestamp', 
                y='power_watts', 
                color='phase',
                title='Power Consumption by Phase'
            )
            return fig
    
    def run(self, port=8050):
        """
        Run interactive dashboard
        """
        self.app.run_server(debug=True, port=port)

# Example usage
if __name__ == '__main__':
    viz = PhaseTaggedPowerViz('power_samples.csv')
    viz.run()