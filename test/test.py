def read_frame(capture):
    capture.grab()
    ret, frame = capture.retrieve()
    print(frame)
    if not ret:
        print('Empty Frame')
        return
    return frame