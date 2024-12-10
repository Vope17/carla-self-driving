import carla
import random
import pygame
import numpy as np
import os
import threading
import weakref
import time
import csv

from evaluate import load_model, automatic_control

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image

import torch

THROTTLE_RATE = 0.6
STEER_RATE = 0.35

log_filename = 'vehicle_data.csv'

def log_vehicle_data(filename, frame, throttle, steer, brake):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame, throttle, steer, brake])

# 初始化pygame
pygame.init()
display = pygame.display.set_mode((640, 480), pygame.HWSURFACE | pygame.DOUBLEBUF)

# 設置目標更新率 (例如，每秒30幀)
clock = pygame.time.Clock()
target_fps = 15
save_interval = 1.0 / target_fps
last_save_time = 0  # 記錄上一次保存影像的時間

reverse = False

class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=1.5, z=2.4)),  # 調整相機位置
            carla.Transform(carla.Location(x=1.5, z=2.4))]
        self.transform_index = 0
        self.sensors = [
            ['sensor.camera.rgb', carla.ColorConverter.Raw, 'Camera RGB'],
            ['sensor.camera.depth', carla.ColorConverter.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', carla.ColorConverter.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', carla.ColorConverter.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', carla.ColorConverter.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', carla.ColorConverter.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            current_time = time.time()
            global last_save_time
            if current_time - last_save_time >= save_interval:
                save_thread = threading.Thread(target=image.save_to_disk, args=('_out/%08d' % image.frame,))
                save_thread.daemon = True  # 設置線程為守護線程
                save_thread.start()
                last_save_time = current_time
                    
                # CSV
                joystick = pygame.joystick.Joystick(0)
                throttle = joystick.get_axis(1) * -THROTTLE_RATE
                steer = joystick.get_axis(0) * STEER_RATE
                brake = joystick.get_axis(2)
                
                log_vehicle_data(log_filename, image.frame, throttle, steer, brake)

class HUD:
    def __init__(self):
        self.dim = (640, 480)
        self.recording_enabled = False

    def notification(self, text):
        print(text)  # 簡單地在控制台打印通知

class VehicleControl:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.control = carla.VehicleControl()
    
    def apply_control(self, throttle, steer, brake, hand_brake=False, reverse=False):
        self.control.throttle = throttle
        self.control.steer = steer
        self.control.brake = brake
        self.control.hand_brake = hand_brake
        self.control.reverse = reverse
        self.vehicle.apply_control(self.control)
    
    def set_autopilot(self, enabled=True):
        self.vehicle.set_autopilot(enabled)

def main():
    # 超參數
    automatic = False
    rotate = False
    THROTTLE = 0.4
    
    # 影像預處理
    SIZE = (192, 256)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(SIZE, interpolation=InterpolationMode.NEAREST),
        #transforms.Grayscale(3),
        lambda x: torch.unsqueeze(x, 0),
        lambda x: x[:,:,:,:].float()
    ])
    
    # pygame init
    pygame.init()
    pygame.joystick.init()

    if automatic:
        model = load_model("MyModel.pt")
    
    steer = 0
    brake = 0
    hand_brake = 0
    reverse = False

    joystick_count = pygame.joystick.get_count()
    if joystick_count < 1:
        print("沒有檢測到手把")
    
    if joystick_count > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
    
    # 各種車車設置
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    hud = HUD()

    try:
        # 連結世界並設置
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        # spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        spawn_index = 4 if not rotate else 6
        spawn_point = spawn_points[spawn_index]

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        
        if vehicle is not None:
            print(f'車輛生成成功，車輛ID: {vehicle.id}')
            
            vehicle_control = VehicleControl(vehicle)
            camera_manager = CameraManager(vehicle, hud)
            camera_manager.set_sensor(0)

            # 主循環
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.JOYBUTTONDOWN:
                        if event.button == 3: # 錄影
                            camera_manager.toggle_recording()
                        if event.button == 1: # 倒車
                            reverse = not reverse

                # 讀取手把輸入
                throttle = THROTTLE
                if joystick_count > 0:
                    steer = joystick.get_axis(0) * STEER_RATE
                    brake = joystick.get_axis(2)
                    hand_brake = joystick.get_button(0)  # 假設按鈕0是手剎
                
                # 代理控制車車(automatic必須是True的時候)
                if automatic and not reverse:
                    # 把錄影圖像轉正
                    frame_matrix = pygame.surfarray.array3d(display)
                    frame_matrix = np.rot90(frame_matrix, 3)
                    frame_matrix = np.fliplr(frame_matrix)
                    if frame_matrix is not None:
                        steer, brake = automatic_control(model, frame_matrix, transform)
                
                vehicle_control.apply_control(throttle, steer, brake, hand_brake, reverse)

                world.tick()
                
                # 渲染相機影像
                camera_manager.render(display)
                
                # 更新 pygame 顯示
                pygame.display.flip()
                
                # 控制螢幕更新率
                clock.tick(target_fps)
                
    finally:
        # 確保在退出前停止錄影
        if camera_manager.recording:
            camera_manager.toggle_recording()
        
        # 銷毀車輛
        if vehicle is not None:
            vehicle.destroy()
            print(f'車輛銷毀，車輛ID: {vehicle.id}')
        if camera_manager.sensor is not None:
            camera_manager.sensor.destroy()

if __name__ == '__main__':
    main()
    pygame.quit()
