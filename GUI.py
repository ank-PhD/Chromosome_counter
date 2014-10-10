from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.utils import platform
from core_app_methods import loop_dir, loop_fle, afterloop
from kivy.clock import Clock, _default_time as time, mainthread
from kivy.factory import Factory
from kivy.properties import ListProperty

from threading import Thread
from time import sleep

MAX_TIME = 1/60.

class MyWidget(BoxLayout):
    progress_bar = ProgressBar(max=1000)
    text_field = TextInput()

    def __init__(self, **kwargs):
        super(MyWidget, self).__init__(**kwargs)
        self.drives_list.adapter.bind(on_selection_change=self.drive_selection_changed)

    def get_win_drives(self):
        if platform == 'win':
            import win32api

            drives = win32api.GetLogicalDriveStrings()
            drives = drives.split('\000')[:-1]

            return drives
        else:    
            return []

    @mainthread
    def append_to_consommables(self, message):
        App.get_running_app().consommables.append(message)

    def update(self, dt):
        print 'updating, or should be'

    def drive_selection_changed(self, *args):
        selected_item = args[0].selection[0].text
        self.file_chooser.path = selected_item

    def post_process(self, dt):
        afterloop(self)

    def process_file(self, dt):
        loop_fle(self.path, self.filename[0].split('\\')[-1], self)
        App.get_running_app().consommables.append("loop done - for real \n")

    def process_folder(self, dt):
        loop_dir(self.path, self)
        App.get_running_app().consommables.append("loop done - for real \n")

    def load(self,  path, filename, Fast, stack_type):
        self.path = path
        self.filename = filename
        self.stack_type = stack_type
        if Fast:
            t_to_add = 'will try to post-process files pre-processed since the previous >>>'
            self.text_field.text = self.text_field.text + t_to_add + '\n'
            Clock.schedule_once(self.post_process, 0)
        else:
            if filename:
                t_to_add = '>>> will try to pre-process file %s at %s'%(filename[0].split('\\')[-1], path)
                self.text_field.text = self.text_field.text + t_to_add + '\n'
                Clock.schedule_once(self.process_file, 0)
            else:
                t_to_add = '>>> will try to pre-process all images at %s'%path
                self.text_field.text = self.text_field.text + t_to_add + '\n'
                Clock.schedule_once(self.process_folder, 0)


class Loader(App):
    consommables = ListProperty([])

    def build(self):
        Clock.schedule_interval(self.consume, 0)
        return MyWidget()

    def consume(self, *args):
        while self.consommables and time() < (Clock.get_time() + MAX_TIME):
            item = self.consommables.pop(0)
            self.root.ids.text_field.text += item

if __name__ == '__main__':
    Loader().run()