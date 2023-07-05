import tkinter
import cv2

class Application(tkinter.Frame):
    def __init__(self, master = None):
        super().__init__(master)
        self.master.geometry("600*400+100+100")
        self.master.title("Window")
        self.canvas = tkinter.Canvas(self.master)
        self.canvas.bind('<Button-1>', self.canvas_click)
        self.canvas.pack(expand = True, fill = tkinter.BOTH)
        self.capture = cv2.VideoCapture(0)

        self.disp_id = None

    def canvas_click(self, event):
        if self.disp_id is None:
            self.disp_image()
        else:
            self.after_cancel(self.disp_id)
            self.disp_id = None
    
    def disp_image(self):
        bool, frame = self.capture.read()
        

def main():
    root = tkinter.Tk()
    app = Application(master = root)
    app.mainloop()

if __name__ == "__main__":
    main()