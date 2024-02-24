import glob
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import pickle

from synch_mode import CarlaSyncMode
from new_controllers import VehiclePIDController
from utils import *

random.seed(78)

class SimEnv(object):
    def __init__(self, 
        visuals=True,
        target_speed = 30,
        max_iter = 4000,
        start_buffer = 10,
        train_freq = 1,
        save_freq = 200,
        start_ep = 0,
        # max_dist_from_waypoint = 20
        max_dist_from_waypoint = 5
    ) -> None:
        self.visuals = visuals
        if self.visuals:
            self._initiate_visuals()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.load_world('Town02_Opt')
        self.world.unload_map_layer(carla.MapLayer.Decals)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        self.world.unload_map_layer(carla.MapLayer.Particles)
        self.world.unload_map_layer(carla.MapLayer.Props)
        self.world.unload_map_layer(carla.MapLayer.StreetLights)
        

        self.spawn_points = self.world.get_map().get_spawn_points()

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprint = self.blueprint_library.find('vehicle.nissan.patrol')
        #self.vehicle_blueprint = self.blueprint_library.find('vehicle.tesla.model3')

        # input these later on as arguments
        self.global_t = 0 # global timestep
        self.target_speed = target_speed # km/h 
        self.max_iter = max_iter
        self.start_buffer = start_buffer
        self.train_freq = train_freq
        self.save_freq = save_freq
        self.start_ep = start_ep

        self.max_dist_from_waypoint = max_dist_from_waypoint
        self.start_train = self.start_ep + self.start_buffer
        
        self.total_rewards = 0
        self.average_rewards_list = []
    
    def _initiate_visuals(self):
        pygame.init()

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()
    
    def create_actors(self):
        self.actor_list = []
        # spawn vehicle at random location
        self.vehicle = self.world.spawn_actor(self.vehicle_blueprint, random.choice(self.spawn_points))
        # vehicle.set_autopilot(True)
        self.actor_list.append(self.vehicle)

        self.camera_rgb = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        self.camera_rgb_vis = self.world.spawn_actor(
            self.blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb_vis)

        self.collision_sensor = self.world.spawn_actor(
            self.blueprint_library.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.actor_list.append(self.collision_sensor)

        # self.speed_controller = PIDLongitudinalController(self.vehicle)
        # self.steer_controller = PIDLateralController(self.vehicle)

        args_lateral = {
            'K_P': 0.5,  # Proportional term. Adjust based on how aggressively you want to correct lateral errors
            'K_D': 0.2,  # Differential term. Use this to counteract overshooting from the P term
            'K_I': 0.07   # Integral term. Useful for eliminating steady-state error
        }

        args_longitudinal = {
            'K_P': 0.5,  # Proportional term. Adjust for the vehicle's response to speed and distance errors
            'K_D': 0.0,  # Differential term. Helps to smooth the response and reduce oscillation
            'K_I': 0.75   # Integral term. Compensates for biases and sustained errors in speed/distance
        }

        self.controller = VehiclePIDController(self.vehicle, args_lateral, args_longitudinal)
        # print("args_lateral =", self.controller._lat_controller._k_i)
        # print("args_longitudinal =", self.controller._lon_controller._k_i)

    
    def reset(self):
        for actor in self.actor_list:
            actor.destroy()
    
    # def generate_episode(self, model, replay_buffer, ep, action_map=None, eval=True):
    def generate_episode(self, ep, action_map=None, eval=True):
        with open(f'deviation_data_episode_{ep}.csv', 'w') as file:
            file.write('LateralDeviation,AngleDeviation\n')

        with CarlaSyncMode(self.world, self.camera_rgb, self.camera_rgb_vis, self.collision_sensor, fps=30) as sync_mode:
            counter = 0

            snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)

            # destroy if there is no data
            if snapshot is None or image_rgb is None:
                print("No data, skipping episode")
                self.reset()
                return None
            
            # Generate route
            self.route_waypoints = []
            self.current_waypoint_index = 0
            self.total_distance = 780 # depending on the town
            #self.total_distance = 5000 # depending on the town

            # Get the initial waypoint based on the vehicle's current location
            current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(), 
                                                        project_to_road=True, lane_type=carla.LaneType.Driving)
            self.route_waypoints.append(current_waypoint)

            for x in range(self.total_distance):
                if x < 650:
                    next_waypoint = current_waypoint.next(0.5)[0]
                else:
                    next_waypoint = current_waypoint.next(0.5)[-1]

                self.route_waypoints.append(next_waypoint)
                current_waypoint = next_waypoint

            image = process_img(image_rgb)
            next_state = image 

            # TO DRAW WAYPOINTS
            for w in self.route_waypoints:
                self.world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                                       persistent_lines=True)
    
            while True:
                if self.visuals:
                    if should_quit():
                        return
                    self.clock.tick_busy_loop(30)
                
                #vehicle_location = self.vehicle.get_location()
                    
                ########################################################
                # Logic to calculate max_d and max_theta
                ########################################################

                # Location of the car
                self.location = self.vehicle.get_location()

                #transform = self.vehicle.get_transform()
                # Keep track of closest waypoint on the route
                waypoint_index = self.current_waypoint_index

                # The purpose of this for loop is to determine 
                # if a vehicle has passed the next waypoint along a predefined route
                for _ in range(len(self.route_waypoints)):
                    # Check if we passed the next waypoint along the route
                    next_waypoint_index = waypoint_index + 1
                    wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]

                    # Computes the dot product between the 2D projection 
                    # of the forward vector of the waypoint (wp.transform.get_forward_vector())
                    # and the vector from the current vehicle location to the waypoint (self.location - wp.transform.location):
                    # positive dot product -> the vectors are pointing at the same direction
                    dot = np.dot(self.vector(wp.transform.get_forward_vector())[:2],self.vector(self.location - wp.transform.location)[:2])
                    
                    # If the dot product is positive, 
                    # it means the vehicle has passed the next waypoint along the route
                    # else, it breaks out of the loop, 
                    # indicating that the current waypoint has not been passed yet:
                    if dot > 0.0:
                        waypoint_index += 1
                    else:
                        break
                
                # waypoint_index: the calculated waypoint index, 
                # which represents the closest waypoint that the vehicle has passed along the planned route
                self.current_waypoint_index = waypoint_index
                # Calculate deviation from center of the lane
                self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
                self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
                
                # CALCULATE d (distance_from_center) and Theta (angle)
                # The result is the distance of the vehicle from the center of the lane:
                self.distance_from_center = self.distance_to_line(self.vector(self.current_waypoint.transform.location),self.vector(self.next_waypoint.transform.location),self.vector(self.location))
                # Get angle difference between closest waypoint and vehicle forward vector
                fwd    = self.vector(self.vehicle.get_velocity())
                wp_fwd = self.vector(self.current_waypoint.transform.rotation.get_forward_vector()) # Return: carla.Vector3D
                self.angle  = self.angle_diff(fwd, wp_fwd)

                with open(f'deviation_data_episode_{ep}.csv', 'a') as file:
                    file.write(f"{self.distance_from_center}, {self.angle}\n")

                # Advance the simulation and wait for the data.
                state = next_state

                counter += 1
                self.global_t += 1


                # action = model.select_action(state, eval=eval)
                # steer = action
                # if action_map is not None:
                #     steer = action_map[action]

                vehicle_loc = self.vehicle.get_location()
                target = self.current_waypoint
                distance_v = self.find_dist_veh(vehicle_loc,target)

                # control_speed = self.speed_controller.run_step(self.target_speed)
                # control_steer = self.steer_controller.run_step(waypoint)
                control = self.controller.run_step(self.target_speed, self.current_waypoint) # control is carla.VehicleControl

                print("The control command that will be applied now =", control)
                self.vehicle.apply_control(control)
                 

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                snapshot, image_rgb, image_rgb_vis, collision = sync_mode.tick(timeout=2.0)

                cos_yaw_diff, dist, collision = get_reward_comp(self.vehicle, self.current_waypoint, collision)
                reward = reward_value(cos_yaw_diff, dist, collision)

                if snapshot is None or image_rgb is None:
                    print("Process ended here")
                    break

                image = process_img(image_rgb)

                done = 1 if collision else 0

                self.total_rewards += reward

                next_state = image

                # replay_buffer.add(state, action, next_state, reward, done)

                # if not eval:
                #     if ep > self.start_train and (self.global_t % self.train_freq) == 0:
                #         model.train(replay_buffer)

                # Draw the display.
                if self.visuals:
                    draw_image(self.display, image_rgb_vis)
                    self.display.blit(
                        self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
                        (8, 10))
                    self.display.blit(
                        self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                        (8, 28))
                    pygame.display.flip()

                if collision == 1 or counter >= self.max_iter or dist > self.max_dist_from_waypoint:
                    print("Episode {} processed".format(ep), counter)
                    break
            
            # if ep % self.save_freq == 0 and ep > 0:
            #     self.save(model, ep)

    def save(self, model, ep):
        if ep % self.save_freq == 0 and ep > self.start_ep:
            avg_reward = self.total_rewards/self.save_freq
            self.average_rewards_list.append(avg_reward)
            self.total_rewards = 0

            model.save('weights/model_ep_{}'.format(ep))

            print("Saved model with average reward =", avg_reward)
    
    def quit(self):
        pygame.quit()

        
    def vector(self, v):
        # The vector method is a utility function that converts a Carla Location, Vector3D, 
        # or Rotation object to a NumPy array for easier manipulation and calculations.
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])

    def angle_diff(self, v0, v1):
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        # If the angular difference is greater than π radians, 
        # it subtracts 2π to bring it within the range [-π, π]:
        if angle > np.pi: angle -= 2 * np.pi
        # If the angular difference is less than or equal to -π radians, 
        # it adds 2π to bring it within the range [-π, π]:
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle
    
    def distance_to_line(self, A, B, p):
        # This method calculates the perpendicular distance from a point p 
        # to a line defined by two points A and B in a 3D space
        num   = np.linalg.norm(np.cross(B - A, A - p)) # calculate cross product 
        denom = np.linalg.norm(B - A) # Euclidean distance between points A and B
        if np.isclose(denom, 0):
            return np.linalg.norm(p - A)
        return num / denom
    
    
    def find_dist_veh(self, vehicle_loc,target):
        dist = math.sqrt( (target.transform.location.x - vehicle_loc.x)**2 + 
                         (target.transform.location.y - vehicle_loc.y)**2 )
    
        return dist
####################################################### End of SimEnv 
    
def get_reward_comp(vehicle, waypoint, collision):
    vehicle_location = vehicle.get_location()
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y

    x_vh = vehicle_location.x
    y_vh = vehicle_location.y

    wp_array = np.array([x_wp, y_wp])
    vh_array = np.array([x_vh, y_vh])

    dist = np.linalg.norm(wp_array - vh_array)

    vh_yaw = correct_yaw(vehicle.get_transform().rotation.yaw)
    wp_yaw = correct_yaw(waypoint.transform.rotation.yaw)
    cos_yaw_diff = np.cos((vh_yaw - wp_yaw)*np.pi/180.)

    collision = 0 if collision is None else 1
    
    return cos_yaw_diff, dist, collision

def reward_value(cos_yaw_diff, dist, collision, lambda_1=1, lambda_2=1, lambda_3=5):
    reward = (lambda_1 * cos_yaw_diff) - (lambda_2 * dist) - (lambda_3 * collision)
    return reward
