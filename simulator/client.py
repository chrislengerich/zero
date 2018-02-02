#!/usr/bin/env python3

# Forked from client_example.py in Carla.

from __future__ import print_function

import argparse
import logging
import random
import sys
import time
import numpy as np
import json

from carla.client import make_carla_client
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from google.protobuf.json_format import MessageToJson

def measurements_to_json(measurements):
    return MessageToJson(measurements)

def save_measurements(frame, episode, measurements):
    import os
    measurements_dir = "_measurements/episode_{:0>3d}/".format(episode)
    if not os.path.exists(measurements_dir):
        os.mkdir(measurements_dir)
    with open("_measurements/episode_{:0>3d}/measurement_{:0>5d}.json".format(episode, frame), "w") as f:
        json_measurements = measurements_to_json(measurements)
        json.dump(json_measurements, f)

def has_car(sensor_data):
    image = sensor_data['CameraSegment']

    try:
        from PIL import Image as PImage
    except ImportError:
        raise RuntimeError('cannot import PIL, make sure pillow package is installed')
    image = PImage.frombytes(
        mode='RGBA',
        size=(image.width, image.height),
        data=image.raw_data,
        decoder_name='raw')
    b, g, r, a = image.split()

    r = np.array(r)
    total_pixels = r.shape[0] * r.shape[1]
    car_pixels = sum(sum(r == 10))
    percent_car = car_pixels / float(total_pixels)
    print(percent_car)
    return percent_car > 0.0075

def run_carla_client(host, port, autopilot_on, save_images_to_disk, image_filename_format, settings_filepath):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 100
    episode_start = 1
    frames_per_episode = 1000

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(host, port) as client:
        print('CarlaClient connected')

        for episode in range(episode_start, number_of_episodes + episode_start):
            # Start a new episode.

            # Load settings from the file.
            with open(settings_filepath, 'r') as fp:
                settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode...')
            client.start_episode(player_start)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Print some of the measurements.
                print_measurements(measurements)

                # Save the images to disk if requested.
                if save_images_to_disk and has_car(sensor_data):
                    for name, image in sensor_data.items():
                        print ("saving %s" % name)
                        image.save_to_disk(image_filename_format.format(episode, name, frame))
                    save_measurements(frame, episode, measurements)

                client.send_control(
                    steer=0, #random.uniform(-1.0, 1.0),
                    throttle=0,
                    brake=False,
                    hand_brake=False,
                    reverse=False)

def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.2f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x / 100, # cm -> m
        pos_y=player_measurements.transform.location.y / 100,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2001,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_false',
        help='save images to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        default=None,
        help='Path to a "CarlaSettings.ini" file')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    while True:
        try:

            run_carla_client(
                host=args.host,
                port=args.port,
                autopilot_on=args.autopilot,
                save_images_to_disk=args.images_to_disk,
                image_filename_format='_images/episode_{:0>3d}/{:s}/image_{:0>5d}.png',
                settings_filepath=args.carla_settings)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
        except Exception as exception:
            logging.exception(exception)
            sys.exit(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
