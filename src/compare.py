def summarize(detections):
    """
    Summarize YOLO detections into simple metrics.
    Returns:
        - n : total detections
        - mean_conf : average confidence of all detections
        - smoke_n : number of 'smoke' detections (case-insensitive)
        - smoke_mean_conf : avg confidence for smoke detections
    """
    if not detections:
        return {
            "n": 0,
            "mean_conf": 0.0,
            "smoke_n": 0,
            "smoke_mean_conf": 0.0
        }

    import statistics as st

    confs = [d["score"] for d in detections]
    smoke_confs = [
        d["score"] for d in detections
        if d["label"].lower() == "smoke"
    ]

    return {
        "n": len(detections),
        "mean_conf": float(st.mean(confs)),
        "smoke_n": len(smoke_confs),
        "smoke_mean_conf": float(st.mean(smoke_confs)) if smoke_confs else 0.0
    }
