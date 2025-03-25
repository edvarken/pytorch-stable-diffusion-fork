import plotly
from plotly import graph_objs as go

def plot_UNET_Time_Breakdown():
    leaf_val1 = 0.319037037
    leaf_val2 = 0.02024222595
    leaf_val3 = 0.6290782995
    color_discrete_sequence = ['#3b99bb', 'rgb(229, 181, 31)', '#ff0000', '#3b99bb', '#3b99bb',
                            '#3b99bb', 'rgb(229, 181, 31)', '#ff0000']
    trace = go.Sunburst(
    values = [0.9683731689, 0.02964275957, 0.001984071498, leaf_val1, leaf_val2, leaf_val3],
    labels = ['UNET', 'VAE Decoding', 'CLIP', 'downsampling', 'middle', 'upsampling'], 
    parents = ['', '', '', 'UNET', 'UNET', 'UNET'], 
    branchvalues='total',
    
    # make the leaf smaller and less visible
    leaf={"opacity": 0.4}, 


    marker={"line": {"width": 2}, "colors":color_discrete_sequence},
    # make UNET values use this color: #3b99bb   
    # make Decoder use this color: #e5b51f
    # make CLIP use this color: #ff0000
    )

    layout = go.Layout(
        margin = go.layout.Margin(t=0, l=0, r=0, b=0),
        annotations=[
        {
        "x": +0.6, 
        "y": 0.46, 
        "ax": +390, 
        "ay": 0, 
        "font": {"size": 25}, 
        "text": "VAE decoding time: 2.8%", 
        "bgcolor": "rgb(229, 181, 31)", 
        "showarrow": True, 
        "bordercolor": "rgb(229, 181, 31)", 
        "borderwidth": 2
        },
        {
        "x": +0.6, 
        "y": 0.497, 
        "ax": +390, 
        "ay": 0, 
        "font": {"size": 25}, 
        "text": "CLIP time: 0.2%", 
        "bgcolor": "rgb(255, 0, 0)", 
        "showarrow": True, 
        "bordercolor": "rgb(255, 0, 0)", 
        "borderwidth": 2
        }
    ]
    )
    # show percentages on the pie chart plot, make them readable in one way
    trace.textinfo = 'label+percent entry' 
    # show percentages OUTSIDE the pie chart plot, make them readable in one way

    trace.insidetextorientation = 'horizontal'
    trace.insidetextfont = dict(size=30)


    fig = go.Figure([trace], layout)
    # Use uniformtext to set a consistent font size
    fig.update_layout(
        title = 'SD1.5 Time Breakdown', 
        # center the title
        title_x = 0.5,
        # title font size
        title_font_size = 32,

        uniformtext=dict(minsize=21),  # Ensures text has a minimum size and can hide overlaps
        # margin=dict(t=0, b=0, l=0, r=0)  # Optional: Adjust margin
        margin=dict(l=0, r=0, t=80, b=50) 
        # move title close to the graph
    )
    fig.show()



def plot_res_vs_atttn():
    color_discrete_sequence = ['#3b99bb', 'rgb(229, 181, 31)', '#ff0000']
    trace = go.Pie(
        values=[0.2778656053, 0.6237245465, 0.0984098482],
        labels=['residual', 'attention', 'other'],
        marker={"line": {"width": 2}, "colors": color_discrete_sequence},
        textinfo='label+percent',
        insidetextorientation='horizontal',
        insidetextfont=dict(size=35)
    )

    layout = go.Layout(
        margin=go.layout.Margin(t=0, l=0, r=0, b=0)
    )

    fig = go.Figure(data=[trace], layout=layout)
    fig.update_layout(
        title= 'UNET Time Breakdown',
        title_x=0.5,
        title_font_size=32,
        uniformtext=dict(minsize=28),
        margin=dict(l=10, r=10, t=80, b=50)
    )
    fig.show()


if __name__ == "__main__":
    plot_UNET_Time_Breakdown()
    # plot_res_vs_atttn()
