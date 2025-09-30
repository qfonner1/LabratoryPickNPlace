# sim_utils.py
import mujoco as mj
from mujoco.glfw import glfw

# Global variables (if needed)
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def keyboard(model, data):
    def handler(window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(model, data)
            mj.mj_forward(model, data)
    return handler

def mouse_button():
    def handler(window, button, act, mods):
        global button_left, button_middle, button_right
        button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        glfw.get_cursor_pos(window)
    return handler

def mouse_move(model, scene, cam):
    def handler(window, xpos, ypos):
        global lastx, lasty, button_left, button_middle, button_right

        dx = xpos - lastx
        dy = ypos - lasty
        lastx = xpos
        lasty = ypos

        if not (button_left or button_middle or button_right):
            return

        width, height = glfw.get_window_size(window)

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)
    return handler

def scroll(model, scene, cam):
    def handler(window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)
    return handler
