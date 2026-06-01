@echo on

for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-ddTHH:mm:ssZ"') do set BUILD_TIME=%%i

docker build --no-cache -t omnivoice:1.0 . --progress=plain --build-arg BUILD_TIME=%BUILD_TIME%

pause