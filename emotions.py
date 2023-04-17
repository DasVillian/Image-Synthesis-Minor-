import gtts
from playsound import playsound
def generate(command,text):
    t1=gtts.gTTS(text)
    t1.save(command+".mp3")


generate("happy","You are happy")
generate("sad","you are sad")
generate("normal","you are neutral")