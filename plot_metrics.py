import argparse
import pandas as pd
import plotly.express as px


def main():
    parser = argparse.ArgumentParser(description="Plot Word2Vec gridsearch metrics with interactive dropdowns")
    parser.add_argument('metrics_json', help="Path to metrics JSON file")
    args = parser.parse_args()

    # Load metrics into DataFrame
    df = pd.read_json(args.metrics_json)
    df['freq_threshold'] = df['freq_threshold'].astype(float)
    df['topk'] = df['topk'].astype(int)

    # Initial axes
    x0, y0 = 'recall', 'precision'
    # Extract possible axis options from DataFrame columns, excluding non-numeric and identifier columns
    exclude_cols = {'model'}
    axis_options = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    # Create initial scatter
    fig = px.scatter(
        df,
        x=x0,
        y=y0,
        color='model',
        symbol='model',
        size='f1',
        hover_data=['topk', 'freq_threshold', 'accuracy'],
        title=f"Gridsearch Metrics: {y0.capitalize()} vs {x0.capitalize()}"
    )

    # Dropdown for filtering by model
    models = df['model'].unique().tolist()
    filter_buttons = [
        dict(
            label='All',
            method='update',
            args=[{'visible': [True]*len(fig.data)},
                  {'title': f"All Models: {y0.capitalize()} vs {x0.capitalize()}"}]
        )
    ]
    for model in models:
        visibility = [trace.name == model for trace in fig.data]
        filter_buttons.append(dict(
            label=model,
            method='update',
            args=[{'visible': visibility},
                  {'title': f"Model: {model} - {y0.capitalize()} vs {x0.capitalize()}"}]
        ))

    # Dropdown for x-axis
    x_buttons = [
        dict(label=f"X: {opt}",
             method='restyle',
             args=[{'x': [df[df.model == trace.name][opt].tolist() for trace in fig.data]},
                   {'xaxis.title.text': opt.capitalize(),
                    'title.text': f"Gridsearch Metrics: {y0.capitalize()} vs {opt.capitalize()}"}]
        )
        for opt in axis_options
    ]
    # Dropdown for y-axis
    y_buttons = [
        dict(label=f"Y: {opt}",
             method='restyle',
             args=[{'y': [df[df.model == trace.name][opt].tolist() for trace in fig.data]},
                   {'yaxis.title.text': opt.capitalize(),
                    'title.text': f"Gridsearch Metrics: {opt.capitalize()} vs {x0.capitalize()}"}]
        )
        for opt in axis_options
    ]

    # Combine all updatemenus with labels via annotations
    fig.update_layout(
        annotations=[
            dict(text="Model:", x=1.05, y=1.15, xref='paper', yref='paper', showarrow=False),
            dict(text="X-axis:", x=0,    y=1.15, xref='paper', yref='paper', showarrow=False),
            dict(text="Y-axis:", x=0.3,  y=1.15, xref='paper', yref='paper', showarrow=False)
        ],
        updatemenus=[
            dict(active=0, buttons=filter_buttons, x=1.15, y=1.15, xanchor='left', yanchor='top'),
            dict(buttons=x_buttons, direction='down', x=0,    y=1.15, xanchor='left', yanchor='top', showactive=True),
            dict(buttons=y_buttons, direction='down', x=0.3, y=1.15, xanchor='left', yanchor='top', showactive=True)
        ],
        # Hide axis titles and tick labels by default
        xaxis=dict(title='', showticklabels=False),
        yaxis=dict(title='', showticklabels=False)
    )

    fig.show()

if __name__ == '__main__':
    main()
