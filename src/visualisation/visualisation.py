from matplotlib import pyplot as plt
from matplotlib.text import Annotation
import torch
from mpl_interactions import zoom_factory
from matplotlib.widgets import CheckButtons, TextBox, Button
import os
import pickle
import numpy as np
from gru.gru import generate_series, default_reservoir, kepler_reservoir


COLOR_PORTFOLIO = np.array(
        ["red", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan",
         "magenta"])
COLOR_MAP = {}


def draw_points(axis, events, last_idx=-1):
    axis.clear()
    init_colors = []
    alphas = []
    color_idx = 0
    x,y = [],[]
    for i in range(len(events)):
        event = events[i]
        x_i, y_i = event.w_2d
        x.append(x_i)
        y.append(y_i)
        alpha = 1. if last_idx == i else 0.5
        alphas.append(alpha)
        #axis.add_artist(Annotation(str(i), (x_i+0.01, y_i+0.01), fontsize=8))
        if hasattr(event, "label") and len(event.label) > 0:
            label = event.label
            if label in COLOR_MAP.keys():
                init_colors.append(COLOR_MAP[label])
            else:
                color = COLOR_PORTFOLIO[color_idx % len(COLOR_PORTFOLIO)]
                COLOR_MAP[label] = color
                init_colors.append(color)
                color_idx += 1
        else:
            init_colors.append("green")

    axis.scatter(x, y, c=init_colors, alpha=alphas, picker=True)


def scatter_plot(ds_name, events, event_path, working_path, ae):
    figure = plt.figure(figsize=(6, 8))
    axis = plt.subplot()
    draw_points(axis, events)

    def onpick(event):
        # step 1: take the index of the dot which was picked
        picked_index = event.ind[0]

        # step 2: save the actual coordinates of the click, so we can position the text label properly
        label_pos_x = event.mouseevent.xdata
        label_pos_y = event.mouseevent.ydata
        print("Picked Index:", picked_index)

        cc_event = events[picked_index]
        plot_curve(cc_event, event_path, axis, events)
        # step 6: force re-draw
        draw_points(axis, events, picked_index)
        axis.figure.canvas.draw_idle()

    #disconnect_zoom = zoom_factory(axis)
    figure.canvas.mpl_connect('pick_event', onpick)
    axis.set_title(ds_name)
    plt.plot()
    plt.savefig(os.path.join(working_path, "{0}.png".format(ds_name)))
    plt.show()


def plot_curve(curve, event_path, main_axis, events):
    figure, axis = plt.subplots(figsize=(10,6))
    figure.subplots_adjust(bottom=0.35)
    original, = axis.plot(curve.timestamps, curve.values, label='Originaldaten')
    lines = [original]
    if hasattr(curve, "gp"):
        gp, = axis.plot(curve.sample_timestamps, curve.gp, label='Gauss-Interpolation')
        lines.append(gp)
    if hasattr(curve, "rnn_prediction"):
        rnn, = axis.plot(curve.sample_timestamps, curve.rnn_prediction, label='RNN-Vorhersage')
        lines.append(rnn)
    if hasattr(curve, "ae_y"):
        ae, = axis.plot(curve.sample_timestamps[1:], curve.ae_y, label='Autoencoder-Rekonstruktion')
        lines.append(ae)
    plt.title('{0}'.format(curve.id))
    axis.legend()
    plt.xlabel('Zeit in Tagen')
    plt.ylabel('Fluss (normalisiert)')
    make_checkbuttons(figure, lines)
    make_labelbox(figure, curve, event_path, main_axis, events)
    plt.show()


def make_checkbuttons(fig, lines):
    # Make checkbuttons with all plotted lines with correct visibility
    rax = fig.add_axes([0.125, 0.1, 0.2, 0.2])
    labels = [str(line.get_label()) for line in lines]
    visibility = [line.get_visible() for line in lines]
    plt.checkbox = CheckButtons(rax, labels, visibility)
    for lbl in plt.checkbox.labels:
        lbl.set_fontsize(7)

    def make_visible(label):
        index = labels.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()

    plt.checkbox.on_clicked(make_visible)


def make_labelbox(fig, event, event_path, axis, events):
    # Add Textbox
    axbox = fig.add_axes([0.6, 0.15, 0.3, 0.075])
    plt.text_box = TextBox(axbox, "Relabel", textalignment="center")

    def label(ev_label):
        event.set_property("label", ev_label)
        with open(os.path.join(event_path, "{0}.pk").format(event.id), "wb") as wpth:
            pickle.dump(event, wpth)
        if ev_label not in COLOR_MAP.keys():
            color_idx = len(COLOR_MAP)
            COLOR_MAP[ev_label] = COLOR_PORTFOLIO[color_idx % len(COLOR_PORTFOLIO)]
        draw_points(axis, events)
        axis.figure.canvas.draw_idle()

    plt.text_box.on_submit(label)
    if hasattr(event, "label"):
        plt.text_box.set_val(event.label)  # Trigger `submit` with the initial string.


def generate_curve(ae, x, y, dataset, ax):
    w_pred = ae.decoder(torch.tensor(np.array([x,y])).float())
    reservoir = kepler_reservoir(w_pred.shape[0])
    if dataset == "cauchy" or dataset == "coffee":
        reservoir = default_reservoir(w_pred.shape[0])
    generated_ts = generate_series(w_pred, 300, reservoir)
    ax.plot(generated_ts)
    ax.figure.canvas.draw_idle()
    plt.draw()


def make_generate_button(ae, dataset, figure):
    figure.subplots_adjust(bottom=0.3)
    def on_button_click(click_event):
        gen_fig, gen_axis = plt.subplots(figsize=(6,4))
        def on_move(event):
            if event.inaxes:
                x, y = event.xdata, event.ydata
                gen_axis.clear()
                generate_curve(ae, x, y, dataset, gen_axis)
                plt.title("Generated Timerseries from cursor coordinates")
        figure.canvas.mpl_connect('motion_notify_event', on_move)
        plt.show()

    button_ax = figure.add_axes([0.3, 0.15, 0.4, 0.075])
    plt.button = Button(button_ax, "Generate Timerseries")
    plt.button.on_clicked(on_button_click)

def plot_png(png_name, plot_path):
    img = plt.imread(plot_path.format(png_name))
    plt.figure()
    png_axis = plt.subplot()
    png_axis.set_title(png_name)
    png_axis.imshow(img)
    #disconnect_zoom = zoom_factory(png_axis)
    plt.plot()
    plt.show()