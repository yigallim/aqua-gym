import pygame
import random
import numpy as np

class Renderer:
    def __init__(self, env):
        self.env = env
        self.screen_width = 1500
        self.screen_height = 900
        self.tank_center_x = self.screen_width // 2
        self.tank_center_y = 400
        self.tank_radius_x = 500
        self.tank_radius_y = 400
        self.min_fish_distance = 60
        self.sound_playing = False

        self.screen = None
        self.font = None
        self.small_font = None
        self.label_font = None
        self.heater_label_font = None
        self.images_loaded = False
        self.pygame_initialized = False
        self.fish_fingerling_img = None
        self.fish_juvenile_img = None
        self.fish_adult_img = None
        self.feeder_img = None
        self.airpump_img = None

        self.fish_positions = []
        self.heat_particles = []
        self.feed_particles = []
        self.bubble_particles = []
        self._initialize_fish_positions()
        self.prev_dissolved_oxygen = 0.6  # Initialize for aeration rate comparison
        
    def _initialize_fish_positions(self):
        self.fish_positions = []
        placed_positions = []

        for _ in range(self.env.initial_fish_count):
            attempts = 0
            max_attempts = 200
            while attempts < max_attempts:
                angle = random.uniform(0, 2 * np.pi)
                r = random.uniform(0, 1) ** 0.5
                x = self.tank_center_x + int(r * (self.tank_radius_x - 50) * np.cos(angle))
                y = self.tank_center_y + int(r * (self.tank_radius_y - 50) * np.sin(angle))
                dx = random.uniform(-1, 1) * 0.5
                dy = random.uniform(-1, 1) * 0.5

                too_close = False
                for px, py, _, _ in placed_positions:
                    dist = ((x - px)**2 + (y - py)**2) ** 0.5
                    if dist < self.min_fish_distance:
                        too_close = True
                        break

                dx_inside = x - self.tank_center_x
                dy_inside = y - self.tank_center_y
                ellipse_equation = (dx_inside**2) / (self.tank_radius_x**2) + (dy_inside**2) / (self.tank_radius_y**2)
                if not too_close and ellipse_equation <= 1.0:
                    self.fish_positions.append([x, y, dx, dy])
                    placed_positions.append([x, y, dx, dy])
                    break

                attempts += 1

            if attempts >= max_attempts:
                x = self.tank_center_x
                y = self.tank_center_y
                dx = random.uniform(-1, 1) * 0.5
                dy = random.uniform(-1, 1) * 0.5
                self.fish_positions.append([x, y, dx, dy])
                placed_positions.append([x, y, dx, dy])

    def render(self):
        if not self.sound_playing:
            try:
                if not pygame.mixer.get_init():
                    pygame.mixer.init()
                pygame.mixer.music.load("assets/tank_sound.mp3")
                pygame.mixer.music.set_volume(1.0)
                pygame.mixer.music.play(-1)
                self.sound_playing = True
            except pygame.error as e:
                print("Audio Error:", e)
                
        if not self.pygame_initialized:
            try:
                pygame.init()
                self.pygame_initialized = True
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Aquaculture Environment")
                self.font = pygame.font.SysFont("Arial", 18)
                self.small_font = pygame.font.SysFont("Arial", 14)
                self.label_font = pygame.font.SysFont("Arial", 18)  # Increased from 16 to 18
                self.heater_label_font = pygame.font.SysFont("Arial", 20)
            except pygame.error as e:
                print(f"Pygame initialization failed: {e}")
                return

        if not self.images_loaded:
            try:
                self.fish_fingerling_img = pygame.image.load("assets/nile_fingerling.png").convert_alpha() 
                self.fish_juvenile_img = pygame.image.load("assets/nile_juvenile.png").convert_alpha()
                self.fish_adult_img = pygame.image.load("assets/nile_adult.png").convert_alpha()
                self.feeder_img = pygame.image.load("assets/feeder.png").convert_alpha()
                self.airpump_img = pygame.image.load("assets/airpump.png").convert_alpha()
                self.images_loaded = True
            except pygame.error as e:
                print(f"Failed Exotic to load images: {e}")
                return

        if not self.pygame_initialized or self.screen is None:
            return

        self.screen.fill((240, 248, 255))

        y = 20
        region_name = getattr(self.env, "region", "unknown").replace("_", " ").title()
        self.screen.blit(self.font.render(f"Region: {region_name}", True, (0, 0, 0)), (20, y))
        y += 25
        self.screen.blit(self.font.render(f"Day: {self.env.day}", True, (0, 0, 0)), (20, y))
        y += 25
        self.screen.blit(self.font.render(f"Biomass: {self.env._compute_total_biomass():.2f}g", True, (0, 0, 0)), (20, y))
        y += 25
        self.screen.blit(self.font.render(f"Fish Count: {self.env._compute_fish_count()}", True, (0, 0, 0)), (20, y))
        y += 25
        self.draw_bar("UIA", self.env.un_ionized_ammonia, 1.4, 20, y, (255, 0, 0))
        y += 30
        self.draw_bar("Temp", self.env.temperature, 40.0, 20, y, (255, 165, 0))

        pygame.draw.ellipse(self.screen, (173, 216, 230), (self.tank_center_x - self.tank_radius_x, self.tank_center_y - self.tank_radius_y, self.tank_radius_x * 2, self.tank_radius_y * 2))
        pygame.draw.ellipse(self.screen, (0, 0, 0), (self.tank_center_x - self.tank_radius_x, self.tank_center_y - self.tank_radius_y, self.tank_radius_x * 2, self.tank_radius_y * 2), 2)

        base_image_width = 70
        base_image_height = 40

        new_positions = []

        for i, (x, y, dx, dy) in enumerate(self.fish_positions):
            repulsion_dx = 0
            repulsion_dy = 0
            for j, (ox, oy, _, _) in enumerate(self.fish_positions):
                if i != j:
                    dist = ((x - ox)**2 + (y - oy)**2) ** 0.5
                    if dist < self.min_fish_distance and dist > 0:
                        force = (self.min_fish_distance - dist) / dist
                        repulsion_dx += force * (x - ox)
                        repulsion_dy += force * (y - oy)

            dx += 0.05 * repulsion_dx
            dy += 0.05 * repulsion_dy

            speed = (dx**2 + dy**2) ** 0.5
            max_speed = 1.0
            if speed > max_speed:
                dx = (dx / speed) * max_speed
                dy = (dy / speed) * max_speed

            new_x = x + dx
            new_y = y + dy

            dx_inside = new_x - self.tank_center_x
            dy_inside = new_y - self.tank_center_y
            margin_x = base_image_width // 2
            margin_y = base_image_height // 2
            effective_radius_x = self.tank_radius_x - margin_x
            effective_radius_y = self.tank_radius_y - margin_y
            ellipse_equation = (dx_inside**2) / (effective_radius_x**2) + (dy_inside**2) / (effective_radius_y**2)

            if ellipse_equation > 1.0:
                normal_x = dx_inside / (effective_radius_x**2)
                normal_y = dy_inside / (effective_radius_y**2)
                normal_length = (normal_x**2 + normal_y**2) ** 0.5
                if normal_length > 0:
                    normal_x /= normal_length
                    normal_y /= normal_length
                dot = dx * normal_x + dy * normal_y
                dx -= 2 * dot * normal_x
                dy -= 2 * dot * normal_y
                new_x = self.tank_center_x + dx_inside * 0.95
                new_y = self.tank_center_y + dy_inside * 0.95

            new_positions.append([new_x, new_y, dx, dy])

        self.fish_positions = new_positions

        for i, fish in enumerate(self.env.fishes):
            x, y, _, _ = self.fish_positions[i]
            weight = fish.weight

            if fish.stage.lower() == "adult":
                min_w, max_w = 50.0, 1000.0
                min_s, max_s = 0.5, 1.5
                if weight <= min_w:
                    scale = min_s
                elif weight >= max_w:
                    scale = max_s
                else:
                    scale = min_s + (max_s - min_s) * (weight - min_w) / (max_w - min_w)
                fish_image = self.fish_adult_img

            elif fish.stage.lower() == "juvenile":
                min_w, max_w = 50.0, 1000.0
                min_s, max_s = 0.5, 1.5
                if weight <= min_w:
                    scale = min_s
                elif weight >= max_w:
                    scale = max_s
                else:
                    scale = min_s + (max_s - min_s) * (weight - min_w) / (max_w - min_w)
                fish_image = self.fish_juvenile_img

            else:
                fl_min = 5.0
                fl_max = getattr(fish, 'to_juvenile_weight', 15.0)
                min_s, max_s = 0.4, 0.6
                if weight <= fl_min:
                    scale = min_s
                elif weight >= fl_max:
                    scale = max_s
                else:
                    scale = min_s + (max_s - min_s) * (weight - fl_min) / (fl_max - fl_min)
                fish_image = self.fish_fingerling_img

            image_width = int(base_image_width * scale)
            image_height = int(base_image_height * scale)
            fish_scaled = pygame.transform.smoothscale(fish_image, (image_width, image_height))
            self.screen.blit(fish_scaled, (int(x - image_width // 2), int(y - image_height // 2)))

            fish_info = f"{fish.stage} | {fish.weight:.1f}g"
            text = self.small_font.render(fish_info, True, (0, 0, 0))
            self.screen.blit(text, (int(x - image_width // 2), int(y + image_height // 2 + 4)))

        # Feeder section
        feeder_x = self.screen_width - 200
        feeder_y = 10
        feeder_width = 160
        feeder_height = 100

        border_padding = 10
        border_x = feeder_x - border_padding
        border_y = feeder_y - border_padding
        feed_text = self.label_font.render(f"Feeding Rate: {self.env.feed_rate_today:.3f}", True, (0, 0, 0))
        total_feed_text = self.label_font.render(f"Total Feed: {self.env.feed_today:.2f}g", True, (0, 0, 0))
        border_width = feeder_width + 2 * border_padding
        border_height = feeder_height + 10 + 20 + 5 + feed_text.get_height() + total_feed_text.get_height() + 5 + 2 * border_padding
        pygame.draw.rect(self.screen, (0, 0, 0), (border_x, border_y, border_width, border_height), 2)

        feeder_scaled = pygame.transform.smoothscale(self.feeder_img, (feeder_width, feeder_height))
        self.screen.blit(feeder_scaled, (feeder_x, feeder_y))

        bar_x = feeder_x + 10
        bar_y = feeder_y + feeder_height + 10
        bar_width = feeder_width - 20
        bar_height = 20
        max_feed_rate = 1.0
        feed_level = min(self.env.feed_rate_today / max_feed_rate, 1.0)
        fill_width = int(bar_width * feed_level)
        if self.env.day > 0 and self.env.feed_rate_today > self.env.feed_rate_yesterday:
            bar_color = (255, 0, 0)
        else:
            bar_color = (0, 100, 0)
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height), 2)
        feed_text = self.label_font.render(f"Feeding Rate: {self.env.feed_rate_today:.3f}", True, (0, 0, 0))
        text_x = bar_x
        text_y = bar_y + bar_height + 5
        self.screen.blit(feed_text, (text_x, text_y))
        total_feed_text = self.label_font.render(f"Total Feed: {self.env.feed_today:.2f}g", True, (0, 0, 0))
        self.screen.blit(total_feed_text, (text_x, text_y + feed_text.get_height() + 5))
        if self.env.day > 0:
            arrow_x = text_x + feed_text.get_width() + 5
            arrow_y_center = text_y + feed_text.get_height() // 2
            arrow_color = bar_color
            if self.env.feed_rate_today > self.env.feed_rate_yesterday:
                pygame.draw.polygon(self.screen, arrow_color, [
                    (arrow_x, arrow_y_center + 4),
                    (arrow_x + 5, arrow_y_center - 8),
                    (arrow_x + 10, arrow_y_center + 4),
                    (arrow_x + 7, arrow_y_center - 2),
                    (arrow_x + 5, arrow_y_center - 8),
                    (arrow_x + 3, arrow_y_center - 2)
                ])
            elif self.env.feed_rate_today < self.env.feed_rate_yesterday:
                pygame.draw.polygon(self.screen, arrow_color, [
                    (arrow_x, arrow_y_center - 4),
                    (arrow_x + 5, arrow_y_center + 8),
                    (arrow_x + 10, arrow_y_center - 4),
                    (arrow_x + 7, arrow_y_center + 2),
                    (arrow_x + 5, arrow_y_center + 8),
                    (arrow_x + 3, arrow_y_center + 2)
                ])

        pipe_color = (0, 0, 0)
        pipe_thickness = 18
        horizontal_length = 200
        pipe_start_x = border_x
        pipe_start_y = feeder_y + feeder_height // 2
        horizontal_end_x = pipe_start_x - horizontal_length
        pygame.draw.line(self.screen, pipe_color, (pipe_start_x, pipe_start_y), (horizontal_end_x, pipe_start_y), pipe_thickness)
        tip_width = 35
        tip_height = 35
        pygame.draw.rect(self.screen, pipe_color, (horizontal_end_x - tip_width // 2, pipe_start_y - tip_height // 2, tip_width, tip_height))

        if self.env.feed_rate_today > 0:
            particle_spawn_rate = self.env.feed_rate_today * 0.5
            if random.random() < particle_spawn_rate:
                particle_x = horizontal_end_x
                particle_y = pipe_start_y
                particle_vx = random.uniform(-0.5, 0.5)
                particle_vy = random.uniform(1, 2)
                particle_radius = random.randint(2, 4)
                particle_life = random.randint(60, 90)
                particle_color = (139, 69, 19)
                self.feed_particles.append([particle_x, particle_y, particle_vx, particle_vy, particle_radius, particle_life, particle_color])

        new_feed_particles = []
        for particle in self.feed_particles:
            x, y, vx, vy, radius, life, color = particle
            x += vx
            y += vy
            life -= 1
            if life > 0 and y < self.tank_center_y + self.tank_radius_y:
                new_feed_particles.append([x, y, vx, vy, radius, life, color])
        self.feed_particles = new_feed_particles

        for particle in self.feed_particles:
            x, y, _, _, radius, life, color = particle
            alpha = int(255 * (life / 90.0))
            surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, color + (alpha,), (radius, radius), radius)
            self.screen.blit(surface, (int(x - radius), int(y - radius)))

        # Air pump section
        airpump_x = self.screen_width - 200
        airpump_y = self.screen_height - 250
        airpump_width = 160
        airpump_height = 100

        border_padding = 30  # Increased from 10 to 15
        border_x = airpump_x - border_padding
        border_y = airpump_y - border_padding
        aeration_text = self.label_font.render(f"Aeration Rate: {self.env.dissolved_oxygen:.3f} mg/L", True, (0, 0, 0))
        border_width = airpump_width + 2 * border_padding
        border_height = airpump_height + 10 + 20 + 5 + aeration_text.get_height() + 10 + 2 * border_padding  # Added +10
        pygame.draw.rect(self.screen, (0, 0, 0), (border_x, border_y, border_width, border_height), 2)

        airpump_scaled = pygame.transform.smoothscale(self.airpump_img, (airpump_width, airpump_height))
        self.screen.blit(airpump_scaled, (airpump_x, airpump_y))

        bar_x = airpump_x + 10
        bar_y = airpump_y + airpump_height + 10
        bar_width = airpump_width - 20
        bar_height = 20
        max_aeration_rate = 1.0
        aeration_level = min(self.env.dissolved_oxygen / max_aeration_rate, 1.0)
        fill_width = int(bar_width * aeration_level)
        if self.env.day > 0 and self.env.dissolved_oxygen > self.prev_dissolved_oxygen:
            bar_color = (255, 0, 0)
        else:
            bar_color = (0, 100, 0)
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, (0, 0, 0), (bar_x, bar_y, bar_width, bar_height), 2)
        aeration_text = self.label_font.render(f"Aeration Rate: {self.env.dissolved_oxygen:.3f} mg/L", True, (0, 0, 0))
        text_x = bar_x -15
        text_y = bar_y + bar_height + 5
        self.screen.blit(aeration_text, (text_x, text_y))
        if self.env.day > 0:
            arrow_x = text_x + aeration_text.get_width() + 5
            arrow_y_center = text_y + aeration_text.get_height() // 2
            arrow_color = bar_color
            if self.env.dissolved_oxygen > self.prev_dissolved_oxygen:
                pygame.draw.polygon(self.screen, arrow_color, [
                    (arrow_x, arrow_y_center + 4),
                    (arrow_x + 5, arrow_y_center - 8),
                    (arrow_x + 10, arrow_y_center + 4),
                    (arrow_x + 7, arrow_y_center - 2),
                    (arrow_x + 5, arrow_y_center - 8),
                    (arrow_x + 3, arrow_y_center - 2)
                ])
            elif self.env.dissolved_oxygen < self.prev_dissolved_oxygen:
                pygame.draw.polygon(self.screen, arrow_color, [
                    (arrow_x, arrow_y_center - 4),
                    (arrow_x + 5, arrow_y_center + 8),
                    (arrow_x + 10, arrow_y_center - 4),
                    (arrow_x + 7, arrow_y_center + 2),
                    (arrow_x + 5, arrow_y_center + 8),
                    (arrow_x + 3, arrow_y_center + 2)
                ])

        pipe_color = (0, 0, 0)
        pipe_thickness = 18
        horizontal_length = 200
        pipe_start_x = border_x
        pipe_start_y = airpump_y + airpump_height // 2
        horizontal_end_x = pipe_start_x - horizontal_length
        pygame.draw.line(self.screen, pipe_color, (pipe_start_x, pipe_start_y), (horizontal_end_x, pipe_start_y), pipe_thickness)
        tip_width = 35
        tip_height = 35
        pygame.draw.rect(self.screen, pipe_color, (horizontal_end_x - tip_width // 2, pipe_start_y - tip_height // 2, tip_width, tip_height))

        if random.random() < 0.3:
            particle_x = horizontal_end_x
            particle_y = pipe_start_y
            particle_vx = random.uniform(-0.2, 0.2)
            particle_vy = random.uniform(-1.5, -0.5)
            particle_radius = random.randint(3, 6)
            particle_life = random.randint(60, 90)
            particle_color = (200, 200, 255)
            self.bubble_particles.append([particle_x, particle_y, particle_vx, particle_vy, particle_radius, particle_life, particle_color])

        new_bubble_particles = []
        for particle in self.bubble_particles:
            x, y, vx, vy, radius, life, color = particle
            x += vx
            y += vy
            life -= 1
            if life > 0 and y > self.tank_center_y - self.tank_radius_y:
                new_bubble_particles.append([x, y, vx, vy, radius, life, color])
        self.bubble_particles = new_bubble_particles

        for particle in self.bubble_particles:
            x, y, _, _, radius, life, color = particle
            alpha = int(255 * (life / 90.0))
            surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, color + (alpha,), (radius, radius), radius)
            self.screen.blit(surface, (int(x - radius), int(y - radius)))

        # Heater section
        heater_x = self.tank_center_x - self.tank_radius_x + 25
        heater_y = self.tank_center_y + self.tank_radius_y + 40
        heater_width = self.tank_radius_x * 2 - 50
        heater_height = 20

        # Calculate temp_heated
        ambient_temp = self.env.temperature_model.get_ambient_temperature()
        temp_setpoint = self.env.temperature  # Assuming current temperature is the setpoint
        temp_heated = max(temp_setpoint - ambient_temp, 0.0)

        # Draw heater
        heater_color = (255, 165, 0) if temp_heated > 0 else (192, 192, 192)
        pygame.draw.rect(self.screen, heater_color, (heater_x, heater_y, heater_width, heater_height))
        pygame.draw.rect(self.screen, (105, 105, 105), (heater_x, heater_y, heater_width, heater_height), 2)
        
        # Draw heater label and temp_heated info
        heater_label = self.heater_label_font.render("Heater", True, (0, 0, 0))
        temp_heated_text = self.label_font.render(f"Heat Added: {temp_heated:.2f}°C", True, (0, 0, 0))
        label_x = heater_x + (heater_width - heater_label.get_width()) // 2
        label_y = heater_y + (heater_height - heater_label.get_height()) // 2
        temp_heated_x = heater_x + (heater_width - temp_heated_text.get_width()) // 2
        temp_heated_y = heater_y - temp_heated_text.get_height() - 5
        self.screen.blit(temp_heated_text, (temp_heated_x, temp_heated_y))
        self.screen.blit(heater_label, (label_x, label_y))

        # Spawn heat particles based on temp_heated
        particle_spawn_rate = temp_heated * 0.1  # Scale particle spawn rate with temp_heated
        if random.random() < particle_spawn_rate:
            particle_x = random.uniform(heater_x, heater_x + heater_width)
            particle_y = heater_y
            particle_vy = random.uniform(-2, -1)
            particle_vx = random.uniform(-0.2, 0.2)
            particle_radius = random.randint(2, 4)
            particle_life = random.randint(30, 60)
            particle_color = (255, random.randint(69, 165), 0)
            self.heat_particles.append([particle_x, particle_y, particle_vx, particle_vy, particle_radius, particle_life, particle_color])

        new_particles = []
        for particle in self.heat_particles:
            x, y, vx, vy, radius, life, color = particle
            x += vx
            y += vy
            life -= 1
            if life > 0 and y > self.tank_center_y - self.tank_radius_y:
                new_particles.append([x, y, vx, vy, radius, life, color])
        self.heat_particles = new_particles

        for particle in self.heat_particles:
            x, y, _, _, radius, life, color = particle
            alpha = int(255 * (life / 60.0))
            surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, color + (alpha,), (radius, radius), radius)
            self.screen.blit(surface, (int(x - radius), int(y - radius)))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        if self.pygame_initialized:
            try:
                pygame.display.flip()
            except pygame.error as e:
                print(f"Pygame display flip failed: {e}")
                self.close()

        self.prev_dissolved_oxygen = self.env.dissolved_oxygen  # Update for next frame

    def draw_bar(self, label, value, max_value, x, y, color):
        if not self.pygame_initialized or self.screen is None:
            return
        width = 150
        filled = int((value / max_value) * width)
        pygame.draw.rect(self.screen, color, (x, y, filled, 20))
        pygame.draw.rect(self.screen, (0, 0, 0), (x, y, width, 20), 2)
        text = self.font.render(f"{label}: {value:.2f}", True, (0, 0, 0))
        self.screen.blit(text, (x + width + 10, y))

    def close(self):
        if self.pygame_initialized:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()  # ✅ Important: shut down the mixer too
                pygame.quit()
            except pygame.error:
                pass
            self.pygame_initialized = False
            self.screen = None
            self.images_loaded = False
            self.sound_playing = False  # 

    def reset(self):
        self.fish_positions = []
        self.heat_particles = []
        self.feed_particles = []
        self.bubble_particles = []
        self._initialize_fish_positions()
        self.prev_dissolved_oxygen = 0.6  # Reset to initial value