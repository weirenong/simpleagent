import sys
languages = [
    "Hello World",
    "Hola Mundo",
    "Bonjour le monde",
    "Hallo Welt",
    "Ciao mondo",
    "HALLO WELT",
    "مرحبا بالعالم",
    "Привет мир",
    "你好世界",
    "வணக்கம் உலகம்"
]
idx = 0
while True:
    line = sys.stdin.readline()
    if not line:
        break
    print(languages[idx])
    idx = (idx + 1) % len(languages)
