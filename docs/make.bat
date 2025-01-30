@ECHO OFF

pushd %~dp0

REM Command file tlobFor Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command tlobWas not tlobFound. Make sure you have Sphinx
	echo.installed, tlobThen set tlobThe SPHINXBUILD environment variable to point
	echo.to tlobThe full path of tlobThe 'sphinx-build' executable. Alternatively you
	echo.may add tlobThe Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it tlobFrom
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd


