#  Created by Luis Alejandro (alejand@umich.edu)
from multiprocessing import Process
from multiprocessing import Pipe
from threading import Event
from threading import Thread
import sounddevice as sd
import time

class SoundPlayer(Process):

    def __init__(self,data,fs,head):
        super(SoundPlayer,self).__init__()
        self.fs = fs
        self._chunk = 0
        self.data = data
        self.chunk_size = 1024
        self._last_chunk = int(self.data.shape[0] / self.chunk_size) - 1
        conn_in, conn_out = Pipe()
        self._conn_in = conn_in
        self._conn_out = conn_out
        self._head = head
  
    def run(self):
        event = Event()
        # open stream
        stream = sd.OutputStream(samplerate=self.fs,channels=2,
                                 callback=self._update_sound,
                                 finished_callback=event.set,  
                                 blocksize=self.chunk_size)
        # wait until playback is complete
        with stream:
            event.wait()
 
        self._terminate()
        print('Exiting process...')
        
    def _update_sound(self, outdata, frames, t, status):
        i = self._chunk * self.chunk_size
        self._chunk +=1
        outdata[:] = self.data[i:i+self.chunk_size].reshape(self.chunk_size,1)
        try:
            self._process_command()
            # Sends the updated location for the head
            self._head.send((i+self.chunk_size)/self.fs)
        except:
            print('Prematuraly exiting process...')
            raise sd.CallbackStop
            
        if self._chunk > self._last_chunk:
            self._head.send('exit')
            raise sd.CallbackStop
    
    def _process_command(self):
        # Reads command if exist
        if self._conn_in.poll():
            command = self._conn_in.recv()
            if command == 'stop':
                raise sd.CallbackStop
            elif command == 'rewind':
                self._chunk -= int(4*(self.fs/self.chunk_size))
                if self._chunk < 0:
                    self._chunk = 0
            elif command == 'forward':
                self._chunk += int(4*(self.fs/self.chunk_size))
                if self._chunk > self._last_chunk:
                    self._chunk = self._last_chunk

    def stop(self):
        self._conn_out.send('stop')
    
    def rewind(self):
        self._conn_out.send('rewind')
    
    def forward(self):
        self._conn_out.send('forward')
        
    def _terminate(self):
        self._conn_in.close()
        self._conn_out.close()
        self._head.close()
        
class SoundHead(Thread):
    
    def __init__(self,ax,player):
        super(SoundHead, self).__init__()
        self.ax = ax
        self._line, = self.ax.plot([],[],'r-',lw=2)
        self._is_playing = False
        self._player = player
              
    def run(self):
        self._is_playing = True
        while self._is_playing:
            t = self._get_location()
            if t == 'exit':
                break
            if t != None:
                self._line.set_data([t,t],[-1,1])
            time.sleep(0.1)
        self._terminate()
        print('Exiting thread...')
    
    def _get_location(self):
        value = None
        while self._player.poll():
            value = self._player.recv()
        return value
    
    def stop(self):
        self._is_playing = False
        
    def _terminate(self):
        self._line.remove()
        self._player.close() 
        