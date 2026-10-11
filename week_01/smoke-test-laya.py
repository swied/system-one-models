from laya import Router
router = Router(preload=True)

result = router.predict(
    state="The build is red on a Friday evening.",
    questions={"deploy": {
        "type": "choice",
        "instructions": "Deploy now or wait?",
        "options": ["deploy", "wait"]}},
)
print(result)
