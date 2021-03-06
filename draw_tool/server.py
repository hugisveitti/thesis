from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
import numpy as np
import json
import cgi
import os
from .run_models import handle_images
from .draw_models_utils import lc_to_sieve

curr_dir = "draw_tool"

data_dir = "data/grid_dir/test/"
rgb_files = os.listdir(os.path.join(data_dir, "rgb"))
print("rgb files",len(rgb_files))
global unchanged_lc
global current_file
unchanged_lc = None
current_file = ""


class Serv(BaseHTTPRequestHandler):

    def do_GET(self):
        
        if self.path == "/":
            file_to_open = open(os.path.join(curr_dir,"index.html")).read()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))
        
        elif self.path.split("/")[1] == "images":
            paths = self.path.split("/")
            print("paths 2",paths[2])
            if len(paths) > 2 and paths[2] != "" and paths[2] in rgb_files:
                fn = paths[2]
            else:
                idx = np.random.randint(len(rgb_files))
                fn = rgb_files[idx]
                print("Using random image")
            print("file", fn)
            with Image.open(os.path.join(data_dir, "rgb",fn)) as f:
                rgb = np.array(f)[:,:,:3]

            with np.load(os.path.join(data_dir, "lc_sieve", fn.split(".")[0] + ".npz")) as f:
                lc = f["arr_0"]
                lc = lc_to_sieve(lc)
            
            d = {
                "rgb":rgb.flatten().tolist(),
                "lc":lc.flatten().tolist()
            }
            global unchanged_lc, current_file
            unchanged_lc = lc
            current_file = fn
            
            self.send_response(200)
            self.end_headers()
      
            self.wfile.write(bytes(json.dumps(d),'utf-8'))
        else:
            self.send_response(404)

    def do_POST(self):
        print(self.path)
        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        length = int(self.headers['content-length'])
        message = json.loads(self.rfile.read(length))
        message['received'] = 'ok'
        
        # send the message back
        self.send_response(200)
        self.end_headers()

        model_name = self.path.split("/")[2]
        global unchanged_lc, current_file
        
        fake_img = handle_images(message, model_name, unchanged_lc, current_file)
        self.wfile.write(bytes(json.dumps((fake_img.flatten()).tolist()),'utf-8'))

def run_server():
    port = 8080
    print(f"Running server on port {port}...")
    httpd = HTTPServer(("localhost", 8080), Serv)
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()