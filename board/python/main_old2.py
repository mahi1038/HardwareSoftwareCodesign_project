from arduino.app_utils import *
import os, time

HERE = os.path.dirname(os.path.abspath(__file__))
done = False


def loop():
    global done
    if done:
        time.sleep(1)
        return
    done = True
    print("HERE:", HERE, flush=True)
    p = os.path.join(HERE, "write_test.txt")
    try:
        with open(p, "a") as f:
            f.write("hello from the board\n")
        print("wrote:", p, "| size:", os.path.getsize(p), flush=True)
    except Exception as e:
        print("write failed:", e, flush=True)
    print("=== done ===", flush=True)


App.run(user_loop=loop)