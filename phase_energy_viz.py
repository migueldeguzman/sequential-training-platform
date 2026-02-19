import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import json

class PhaseEnergyVisualizer:
    def __init__(self, data_path):
        """
        Initialize the visualizer with power sampling data
        
        Args:
            data_path (str): Path to JSON file with power sampling data
        """
        with open(data_path, 'r') as f:
            self.power_samples = json.load(f)
        
        self.df = self._prepare_dataframe()
        self.app = self._create_dash_app()
    
    def _prepare_dataframe(self):
        """
        Convert power samples to a pandas DataFrame for visualization
        
        Returns:
            pd.DataFrame: Processed power sampling data
        """
        records = []
        for sample in self.power_samples:
            records.append({
                'phase': sample['phase'],
                'power_watts': sample['power_watts'],
                'phase_confidence': sample['phase_confidence'],
                'timestamp': sample.get('timestamp')
            })
        
        return pd.DataFrame(records)
    
    def _create_dash_app(self):
        """
        Create Dash application for energy phase visualization
        
        Returns:
            dash.Dash: Configured Dash application
        """
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1('SDB Energy Profiler: Inference Phase Power Analysis'),
            
            dcc.Graph(id='phase-power-scatter'),
            dcc.Graph(id='phase-power-boxplot'),
            
            dcc.Dropdown(
                id='phase-selector',
                options=[{'label': phase, 'value': phase} 
                         for phase in self.df['phase'].unique()],
                multi=True,
                placeholder='Select Inference Phases'
            )
        ])
        
        @app.callback(
            [Output('phase-power-scatter', 'figure'),
             Output('phase-power-boxplot', 'figure')],
            [Input('phase-selector', 'value')]
        )
        def update_graphs(selected_phases):
            if not selected_phases:
                selected_phases = self.df['phase'].unique()
            
            filtered_df = self.df[self.df['phase'].isin(selected_phases)]
            
            scatter = px.scatter(
                filtered_df, 
                x='timestamp', 
                y='power_watts', 
                color='phase',
                title='Power Consumption by Inference Phase',
                labels={'power_watts': 'Power (Watts)', 'timestamp': 'Timestamp'}
            )
            
            boxplot = px.box(
                filtered_df, 
                x='phase', 
                y='power_watts',
                title='Power Distribution Across Inference Phases',
                labels={'power_watts': 'Power (Watts)', 'phase': 'Inference Phase'}
            )
            
            return scatter, boxplot
        
        return app
    
    def run(self, port=8050):
        """
        Run the Dash application
        
        Args:
            port (int): Port to run the application on
        """
        self.app.run_server(debug=True, port=port)

def main():
    visualizer = PhaseEnergyVisualizer(
        '/Users/miguelitodeguzman/Projects/AI-projects/SDB/ml-dashboard/power_samples.json'
    )
    visualizer.run()

if __name__ == '__main__':
    main()