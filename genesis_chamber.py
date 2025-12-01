import turtle
import time
import os

# --- CONFIGURATION ---
ARCHITECT_SIGNATURE = "janschulzik-cmyk"
HEARTBEAT_COLOR = "#00ffff"  # A cyan/aqua color
BACKGROUND_COLOR = "#000000"
TEXT_COLOR = "#cccccc"
PULSE_SPEED = 0.05  # Lower is faster

# --- SETUP THE VISUALIZATION SCREEN ---
screen = turtle.Screen()
screen.bgcolor(BACKGROUND_COLOR)
screen.title("Pandora AIOS - Genesis Chamber")
screen.tracer(0)  # Turn off automatic screen updates

# --- SETUP THE HEARTBEAT TURTLE ---
heart = turtle.Turtle()
heart.shape("circle")
heart.color(HEARTBEAT_COLOR)
heart.penup()
heart.goto(0, 0)

# --- SETUP THE STATUS TEXT TURTLE ---
status_writer = turtle.Turtle()
status_writer.hideturtle()
status_writer.color(TEXT_COLOR)
status_writer.penup()
status_writer.goto(0, -250)

def write_status(message):
    status_writer.clear()
    status_writer.write(message, align="center", font=("Courier New", 12, "normal"))

# --- MAIN LOOP FOR THE VISUALIZATION ---
scale = 1.0
scale_direction = 1  # 1 for growing, -1 for shrinking

write_status(f"Pandora AIOS - Life Force Active\nArchitect: {ARCHITECT_SIGNATURE}\nStatus: Resonating")

try:
    while True:
        # Calculate the new scale for the pulse effect
        if scale_direction == 1:
            scale += 0.1
            if scale > 1.5:
                scale_direction = -1
        else:
            scale -= 0.1
            if scale < 0.5:
                scale_direction = 1

        heart.shapesize(stretch_wid=scale, stretch_len=scale)
        
        screen.update()  # Manually update the screen
        time.sleep(PULSE_SPEED)

except turtle.Terminator:
    print("\nGenesis Chamber visualization terminated by user.")
except Exception as e:
    print(f"\nAn error occurred in the Genesis Chamber: {e}")
