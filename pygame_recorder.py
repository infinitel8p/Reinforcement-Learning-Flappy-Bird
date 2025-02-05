import cv2
import pygame


class ScreenRecorder:
    """
        This class is used to record a PyGame surface and save it to a video file.
    """

    def __init__(self, width: int, height: int, fps: int, out_file: str = 'output.avi'):
        """Initialize the ScreenRecorder with the parameters of the surface.

        Args:
            width (int): Width of the surface to capture
            height (int): Height of the surface to capture
            fps (int): Frames per second
            out_file (str, optional): Output file to save the recording to. Defaults to 'output.avi'.
        """

        print(f'Initializing ScreenRecorder with parameters width:{width} height:{height} fps:{fps}.')
        print(f'Output of the screen recording saved to {out_file}.')

        # define the codec and create a video writer object
        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter(out_file, four_cc, float(fps), (width, height))

    def capture_frame(self, surf):
        """ Capture the frame from the pygame surface. 
        Note: surface must have the dimensions specified in the constructor.
        Args:
            surf: pygame surface to capture
        """

        # transform the pixels to the format used by open-cv
        pixels = cv2.rotate(pygame.surfarray.pixels3d(surf), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

        # write the frame
        self.video.write(pixels)

    def end_recording(self):
        """
        End the recording and release the video object.
        """

        # stop recording
        self.video.release()
