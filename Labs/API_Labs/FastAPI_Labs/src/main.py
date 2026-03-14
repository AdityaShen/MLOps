from fastapi import FastAPI, status, HTTPException
from data import DigitData, DigitBatch, DigitResponse, BatchResponse
from predict import predict_digit

app = FastAPI()


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy", "message": "Digit Classification API"}


@app.post("/predict", response_model=DigitResponse)
async def predict(digit_features: DigitData):
    try:
        if len(digit_features.pixels) != 64:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 64 pixel values, got {len(digit_features.pixels)}"
            )
        prediction = predict_digit(digit_features)
        return DigitResponse(prediction=prediction)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchResponse)
async def batch_predict(batch: DigitBatch):
    try:
        predictions = []
        for sample in batch.samples:
            if len(sample.pixels) != 64:
                raise HTTPException(
                    status_code=400,
                    detail=f"Each sample must have 64 pixel values"
                )
            pred = predict_digit(sample)
            predictions.append(pred)
        return BatchResponse(predictions=predictions)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_info")
async def model_info():
    return {
        "model": "KNeighborsClassifier",
        "dataset": "Digits (Handwritten Digit Recognition)",
        "features": "64 pixel intensities from 8x8 images",
        "classes": "10 (digits 0-9)",
        "task": "Multiclass digit classification",
    }
