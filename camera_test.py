import sensor, image, time, lcd

lcd.init()
lcd.clear(lcd.RED)

sensor.reset()                      # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565) # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)   # Set frame size to QVGA (320x240)
sensor.skip_frames(time = 2000)     # Wait for settings take effect.
clock = time.clock()                # Create a clock object to track the FPS.

while(True):
    clock.tick()                    # Update the FPS clock.
    img = sensor.snapshot()         # Take a picture and return the image.
    lcd.display(img)                # Display image on lcd.
    print(clock.fps())              # Note: CanMV Cam runs about half as fast when connected
                                    # to the IDE. The FPS should increase once disconnected.
