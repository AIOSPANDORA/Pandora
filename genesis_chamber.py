"""
Genesis Chamber - Visual Heartbeat/Life Force Indicator for Pandora AIOS

This module provides a pulsing circle animation using Python's turtle graphics
to visually represent the life force of Pandora AIOS.

The visualization displays:
- A pulsing cyan circle that grows and shrinks
- Status text showing "Pandora AIOS - Life Force Active"
- Architect signature attribution
"""

import turtle

# ============================================================================
# CONFIGURATION
# ============================================================================

ARCHITECT_SIGNATURE = "janschulzik-cmyk"
HEARTBEAT_COLOR = "#00ffff"  # Cyan/Aqua
BACKGROUND_COLOR = "black"
STATUS_TEXT = "Pandora AIOS - Life Force Active"

# Animation parameters
MIN_RADIUS = 30
MAX_RADIUS = 80
PULSE_STEP = 2
ANIMATION_DELAY = 30  # milliseconds


def create_genesis_chamber():
    """
    Create and run the Genesis Chamber visualization.

    Sets up a turtle graphics window with a black background,
    displays a pulsing cyan circle animation, and shows status text.
    """
    # Set up the screen
    screen = turtle.Screen()
    screen.title("Genesis Chamber - Pandora AIOS")
    screen.bgcolor(BACKGROUND_COLOR)
    screen.setup(width=600, height=500)
    screen.tracer(0)  # Disable auto-refresh for smoother animation

    # Create the heartbeat turtle
    heartbeat = turtle.Turtle()
    heartbeat.hideturtle()
    heartbeat.speed(0)
    heartbeat.color(HEARTBEAT_COLOR)
    heartbeat.pensize(3)

    # Create the status text turtle
    status = turtle.Turtle()
    status.hideturtle()
    status.penup()
    status.color(HEARTBEAT_COLOR)
    status.goto(0, -150)
    status.write(STATUS_TEXT, align="center", font=("Arial", 14, "bold"))

    # Create the signature text turtle
    signature = turtle.Turtle()
    signature.hideturtle()
    signature.penup()
    signature.color(HEARTBEAT_COLOR)
    signature.goto(0, -180)
    signature.write(
        f"Architect: {ARCHITECT_SIGNATURE}",
        align="center",
        font=("Arial", 10, "normal"),
    )

    # Animation state
    state = {"radius": MIN_RADIUS, "growing": True}

    def draw_circle(radius):
        """Draw a circle with the given radius."""
        heartbeat.clear()
        heartbeat.penup()
        heartbeat.goto(0, -radius)
        heartbeat.pendown()
        heartbeat.circle(radius)

    def pulse():
        """Animate the pulsing circle."""
        # Update radius based on direction
        if state["growing"]:
            state["radius"] += PULSE_STEP
            if state["radius"] >= MAX_RADIUS:
                state["growing"] = False
        else:
            state["radius"] -= PULSE_STEP
            if state["radius"] <= MIN_RADIUS:
                state["growing"] = True

        # Draw the circle at the current radius
        draw_circle(state["radius"])
        screen.update()

        # Schedule the next animation frame
        screen.ontimer(pulse, ANIMATION_DELAY)

    # Start the animation
    pulse()

    # Start the event loop to keep the window responsive
    screen.mainloop()


if __name__ == "__main__":
    create_genesis_chamber()
