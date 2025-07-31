rm -rf build
mkdir build && cd build
cmake ..
make
cd ..
mv build/hvcpp.cpython-* venv/lib/python3.12/site-packages/
python hv_test.py
