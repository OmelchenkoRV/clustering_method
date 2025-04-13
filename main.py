from fastapi import FastAPI
from app.models.request import CalculationRequest
from app.services.calculation import compute_service_stats, calculate_package_statistics
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)


@app.post("/compute")
def compute(request: CalculationRequest):
    logging.info("Computation request received")
    service_stats = compute_service_stats(request.price, request.support_cost, request.daily_orders, request.n_days)
    package_statistics = calculate_package_statistics(request.price, request.support_cost, request.daily_orders,
                                                      request.n_days)
    logging.info(f"Computation completed. Result: {service_stats}, Package statistics: {package_statistics}")
    return {"success": True, "service_stats": service_stats, "package_statistics": package_statistics}
