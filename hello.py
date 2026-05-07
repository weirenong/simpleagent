import tkinter as tk
import random

def hello_random():
    languages = [
        "Hello, World!",
        "Bonjour le monde!",
        "Hola, Mundo!",
        "Ciao, Mondo!",
        "Hallo, Welt!"
    ]
    # Print a random greeting each time the button is pressed
    print(random.choice(languages))

def bye_random():
    languages = [
        "Bye Bye World!",
        "Adieu le monde!",
        "Adiós, Mundo!",
        "Farewell, Mondo!",
        "Auf Wieder Welt!",
        "Hallo Welt!",
        "Bye, World!",
        "你好，世界！"   # <-- Chinese addition
    ]
    # Print a random goodbye message each time the button is pressed
    print(random.choice(languages))

def quit_app():
    root.quit()

def good_morning_random():
    languages = [
        "Good morning!",
        "Guten Morgen!",
        "Bonjour le matin!",
        "Guten Morgen!",
        "Hallo, Morgen!"
    ]
    # Print a random good‑morning greeting each time the button is pressed
    print(random.choice(languages))

root = tk.Tk()

hello_btn = tk.Button(root, text="Say Hello", command=hello_random)
bye_btn   = tk.Button(root, text="Bye Bye World", command=bye_random)
good_btn  = tk.Button(root, text="Good Morning", command=good_morning_random)
quit_btn  = tk.Button(root, text="Quit", command=quit_app)

hello_btn.pack(pady=5)
bye_btn.pack(pady=5)
good_btn.pack(pady=5)
quit_btn.pack(pady=5)

root.mainloop()
