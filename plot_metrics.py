import argparse
import pandas as pd
import plotly.express as px


def main():
    parser = argparse.ArgumentParser(description="Plot Word2Vec gridsearch metrics with interactive dropdowns")
    parser.add_argument('metrics_json', help="Path to metrics JSON file")
    args = parser.parse_args()

    # Load metrics into DataFrame
    df = pd.read_json(args.metrics_json)

    # Try casting types only if columns exist
    # if 'freq_threshold' in df.columns:
    #     df['freq_threshold'] = df['freq_threshold'].astype(float)
    # if 'topk' in df.columns:
    #     df['topk'] = df['topk'].astype(int)
    # if 'cos' in df.columns:
    #     df['cos'] = df['cos'].astype(float)

    # Exclude known non-numeric, identifier-like columns
    exclude_cols = {'model'}
    axis_options = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    # Default x and y axes — use any available numeric columns
    x0 = 'recall' if 'recall' in df.columns else axis_options[0]
    y0 = 'precision' if 'precision' in df.columns else axis_options[1]

    # Choose hover columns based on availability
    default_hover = ['topk', 'freq_threshold', 'cos', 'accuracy']
    hover_cols = [col for col in default_hover if col in df.columns]

    # Initial figure
    fig = px.scatter(
        df,
        x=x0,
        y=y0,
        color='model' if 'model' in df.columns else None,
        symbol='model' if 'model' in df.columns else None,
        size='f1' if 'f1' in df.columns else None,
        hover_data=hover_cols,
        title=f"Gridsearch Metrics: {y0.capitalize()} vs {x0.capitalize()}"
    )

    # Dropdown for filtering by model
    if 'model' in df.columns:
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
    else:
        filter_buttons = []

    # Axis dropdowns
    x_buttons = [
        dict(label=f"X: {opt}",
             method='restyle',
             args=[{'x': [df[df.model == trace.name][opt].tolist()
                          if 'model' in df.columns else df[opt].tolist()
                          for trace in fig.data]},
                   {'xaxis.title.text': opt.capitalize(),
                    'title.text': f"Gridsearch Metrics: {y0.capitalize()} vs {opt.capitalize()}"}]
        )
        for opt in axis_options
    ]

    y_buttons = [
        dict(label=f"Y: {opt}",
             method='restyle',
             args=[{'y': [df[df.model == trace.name][opt].tolist()
                          if 'model' in df.columns else df[opt].tolist()
                          for trace in fig.data]},
                   {'yaxis.title.text': opt.capitalize(),
                    'title.text': f"Gridsearch Metrics: {opt.capitalize()} vs {x0.capitalize()}"}]
        )
        for opt in axis_options
    ]

    # Assemble layout with conditional annotations and menus
    annotations = [
        dict(text="X-axis:", x=0, y=1.15, xref='paper', yref='paper', showarrow=False),
        dict(text="Y-axis:", x=0.3, y=1.15, xref='paper', yref='paper', showarrow=False)
    ]
    if 'model' in df.columns:
        annotations.insert(0, dict(text="Model:", x=1.05, y=1.15, xref='paper', yref='paper', showarrow=False))

    updatemenus = [
        dict(buttons=x_buttons, direction='down', x=0, y=1.15, xanchor='left', yanchor='top', showactive=True),
        dict(buttons=y_buttons, direction='down', x=0.3, y=1.15, xanchor='left', yanchor='top', showactive=True)
    ]
    if filter_buttons:
        updatemenus.insert(0, dict(active=0, buttons=filter_buttons, x=1.15, y=1.15, xanchor='left', yanchor='top'))

    fig.update_layout(
        annotations=annotations,
        updatemenus=updatemenus,
        xaxis=dict(title='', showticklabels=False),
        yaxis=dict(title='', showticklabels=False)
    )

    fig.show()


if __name__ == '__main__':
    main()