import sys

import stdio_world

# this makes `import wit_world` succeed when we `import execenv`
stdio_world.__name__ = 'wit_world'
sys.modules['wit_world'] = stdio_world

import execenv

ee = execenv.WitWorld()

# stdio_world needs access to ee because it will call ee.execute_warp_message on it when
# receiving such messages from stdin
stdio_world.ee = ee

stdio_world._start_event_loop(sys.stdin.detach(), sys.stdout.detach())
