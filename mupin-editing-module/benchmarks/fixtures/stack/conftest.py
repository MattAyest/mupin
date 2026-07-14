from hypothesis import HealthCheck, settings

settings.register_profile(
    "sandbox",
    max_examples=50,
    deadline=5000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("sandbox")
