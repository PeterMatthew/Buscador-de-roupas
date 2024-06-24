from ultralytics import YOLO, settings

def main():
    settings.update({'datasets_dir': 'datasets', 'weights_dir': 'weights', 'runs_dir': 'runs'})

    model = YOLO("yolov8l.pt")

    model.train(data="config.yaml", epochs=20, batch=4, device=0)
    metrics = model.val()
    path = model.export()

if __name__ == "__main__":
    main()
