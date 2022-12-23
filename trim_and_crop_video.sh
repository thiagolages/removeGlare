# https://stackoverflow.com/questions/18444194/cutting-the-videos-based-on-start-and-end-time-using-ffmpeg
# crop video example (cut rectangle in video window):
# ffmpeg -ss 00:01:00 -to 00:02:00 -i input.mp4 -c copy output.mp4

# Explanation of the command:

# -i: This specifies the input file. In that case, it is (input.mp4).
# -ss: Used with -i, this seeks in the input file (input.mp4) to position.
# 00:01:00: This is the time your trimmed video will start with.
# -to: This specifies when to stop the trim.
# 00:02:00: This is the time your trimmed video will end at.
# -c copy: This is an option to trim via stream copy. (NB: Very fast)

# Example used by me:
# ffmpeg -ss 00:03:00 -to 00:04:30 -i videos/02.mkv -c copy videos/02_trimmed.mkv


# https://video.stackexchange.com/questions/4563/how-can-i-crop-a-video-with-ffmpeg
# crop video example (cut rectangle in video window):
# ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4

x=1143
y=663
w=446
h=331

ffmpeg -i videos/02_trimmed.mkv -filter:v "crop=$w:$h:$x:$y" videos/02_final.mkv -c copy