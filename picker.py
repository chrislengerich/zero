import matplotlib.pyplot as plt
import pprint

from loader import CarlaDataset

def line_picker(unused_line, mouseevent):
    props = dict(ind=0, pickx=mouseevent.xdata, picky=mouseevent.ydata)
    return True, props

def onpick2(event):
    # import matplotlib.backend_bases.PickEvent
    values = { "x": event.mouseevent.xdata, "y": event.mouseevent.ydata }
    print values

def pick_points(point):
    fig, ax1 = plt.subplots(1)
    ax1.set_title('Pick a location in the image')
    ax1.imshow(point['rgb'], picker=line_picker)

    fig.canvas.mpl_connect('pick_event', onpick2)
    plt.show()

if __name__ == "__main__":
    locations = []
    data = CarlaDataset("/home/ubuntu/zero/data/val_carla_single_car")
    for i in range(50):
        pick_points(data[i])
        locations.append([data.closest_cars(i)[0]["world_position"]])
        print locations[-1]


