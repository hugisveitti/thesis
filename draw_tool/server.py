from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
import numpy as np
import json
import cgi
import os
from .utils import handle_images

curr_dir = "draw_tool"

data_dir = "data/grid_dir/val/"
rgb_files = os.listdir(os.path.join(data_dir,"rgb"))
print("rgb files",len(rgb_files))

class Serv(BaseHTTPRequestHandler):

    def do_GET(self):
        idx = np.random.randint(len(rgb_files))
        fn = rgb_files[idx]
        print("fn",fn)

        if self.path == "/":
            file_to_open = open(os.path.join(curr_dir,"index.html")).read()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))

        elif self.path == "/images":
            with Image.open(os.path.join(data_dir, "rgb",fn)) as f:
                rgb = np.array(f)[:,:,:3]

            with Image.open(os.path.join(data_dir, "lc", fn)) as f:
                lc = np.array(f)[:,:,:3]
            
            d = {
                "rgb":rgb.flatten().tolist(),
                "lc":lc.flatten().tolist()
            }
            
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
        print("message",message.keys())
        message['received'] = 'ok'
        
        # send the message back
        self.send_response(200)
        self.end_headers()
        # self.wfile.write(bytes(json.dumps(message), "utf-8"))

        use_inpaint = self.path.split("/")[2] == "inpaint"
        fake_img = handle_images(message, use_inpaint)
        self.wfile.write(bytes(json.dumps((fake_img.flatten()).tolist()),'utf-8'))

def run_server():
    port = 8080
    print(f"Running server on port {port}...")
    httpd = HTTPServer(("localhost", 8080), Serv)
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()