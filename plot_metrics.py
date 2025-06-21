import argparse
import pandas as pd
import plotly.express as px

def main():
    parser = argparse.ArgumentParser(
        description="Plot Word2Vec gridsearch metrics with interactive dropdowns"
    )
    parser.add_argument('metrics_json', help="Path to metrics JSON file")
    args = parser.parse_args()

    # 1) Load JSON unambiguously as a list of records
    df = pd.read_json(args.metrics_json, orient='records')

    # 2) Coerce all your important columns to real numerics
    for col in ['recall', 'precision', 'f1', 'accuracy', 'freq_threshold', 'topk', 'cos']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='raise')

    # 3) Figure out which columns are truly numeric (excluding identifiers)
    exclude_cols = {'model'}
    axis_options = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    # 4) Pick defaults from only those numeric columns
    x0 = 'recall' if 'recall' in axis_options else axis_options[0]
    y0 = 'precision' if 'precision' in axis_options else axis_options[1]

    # 5) Build a hover‐data list that actually includes recall & precision
    hover_cols = [c for c in ['recall','precision','topk','freq_threshold','cos','accuracy']
                  if c in df.columns]

    # 6) Create the scatter
    fig = px.scatter(
        df,
        x=x0,
        y=y0,
        color='model'   if 'model' in df.columns else None,
        symbol='model'  if 'model' in df.columns else None,
        size='f1'       if 'f1'   in df.columns else None,
        hover_data=hover_cols,
        title=f"Gridsearch Metrics: {y0.capitalize()} vs {x0.capitalize()}"
    )

    # 7) (unchanged) model-filter dropdown
    if 'model' in df.columns:
        models = df['model'].unique().tolist()
        filter_buttons = [{
            'label':'All','method':'update',
            'args':[{'visible':[True]*len(fig.data)},
                    {'title':f"All Models: {y0.capitalize()} vs {x0.capitalize()}"}]
        }]
        for m in models:
            vis = [trace.name==m for trace in fig.data]
            filter_buttons.append({
                'label':m,'method':'update',
                'args':[{'visible':vis},
                        {'title':f"Model: {m} — {y0.capitalize()} vs {x0.capitalize()}"}]
            })
    else:
        filter_buttons = []

    # 8) Axis-changing dropdowns (unchanged)
    x_buttons = [
        dict(
            label=f"X: {opt}",
            method='restyle',
            args=[{'x': [
                df[df.model==trace.name][opt].tolist()
                if 'model' in df.columns else df[opt].tolist()
                for trace in fig.data
            ]},
            {'xaxis.title.text':opt.capitalize(),
             'title.text':f"Gridsearch Metrics: {y0.capitalize()} vs {opt.capitalize()}"}]
        )
        for opt in axis_options
    ]
    y_buttons = [
        dict(
            label=f"Y: {opt}",
            method='restyle',
            args=[{'y': [
                df[df.model==trace.name][opt].tolist()
                if 'model' in df.columns else df[opt].tolist()
                for trace in fig.data
            ]},
            {'yaxis.title.text':opt.capitalize(),
             'title.text':f"Gridsearch Metrics: {opt.capitalize()} vs {x0.capitalize()}"}]
        )
        for opt in axis_options
    ]

    # 9) Layout: show tick labels & axis titles
    annotations = [
        dict(text="X-axis:", x=0,   y=1.15, xref='paper', yref='paper', showarrow=False),
        dict(text="Y-axis:", x=0.3, y=1.15, xref='paper', yref='paper', showarrow=False)
    ]
    if 'model' in df.columns:
        annotations.insert(0, dict(
            text="Model:", x=1.05, y=1.15, xref='paper', yref='paper', showarrow=False
        ))

    updatemenus = [
        dict(buttons=filter_buttons, direction='down',
             x=1.15, y=1.15, xanchor='left', yanchor='top', showactive=True)
    ] if filter_buttons else []
    updatemenus += [
        dict(buttons=x_buttons, direction='down',
             x=0,    y=1.15, xanchor='left', yanchor='top', showactive=True),
        dict(buttons=y_buttons, direction='down',
             x=0.3,  y=1.15, xanchor='left', yanchor='top', showactive=True)
    ]

    fig.update_layout(
        annotations=annotations,
        updatemenus=updatemenus,
        xaxis=dict(title=x0.capitalize(), showticklabels=True),
        yaxis=dict(title=y0.capitalize(), showticklabels=True),
    )

    # 10) (Optional but foolproof) lock the axes exactly to your data range
    fig.update_xaxes(range=[df[x0].min(), df[x0].max()])
    fig.update_yaxes(range=[df[y0].min(), df[y0].max()])

    fig.show()

if __name__ == '__main__':
    main()
