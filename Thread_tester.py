__author__ = 'ank'
from kivy.app import App
from kivy.clock import Clock, _default_time as time  # ok, no better way to use the same clock as kivy, hmm
from kivy.factory import Factory
from kivy.uix.button import Button
from kivy.properties import ListProperty

from threading import Thread
from time import sleep

MAX_TIME = 1/60.

class MyButton(Button):
    def on_press(self, *args):
        Thread(target=self.worker).start()

    def worker(self):
        sleep(5) # blocking operation
        App.get_running_app().consommables.append("done")

class PubConApp(App):
    consommables = ListProperty([])

    def build(self):
        Clock.schedule_interval(self.consume, 0)

    def consume(self, *args):
        while self.consommables and time() < (Clock.get_time() + MAX_TIME):
            item = self.consommables.pop(0)  # i want the first one
            label = Factory.MyLabel(text=item)
            self.root.ids.target.add_widget(label)

if __name__ == '__main__':
    PubConApp().run()