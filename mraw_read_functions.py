import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import numba as nb

def clean_cih(data_cih):
    '''Cleans invalid characters from a cih file.
    Also avoids garbage at the end of the file.
    Writes the cleaned CIH file with a ".xml" suffix.
    Input
    data_cih: Path object for the CIH file to be cleaned
    Return
    cleaned_cih: Path object for XML file with cleaned CIH data
    '''
    cleaned_cih = data_cih.parent.joinpath(data_cih.stem + '.xml')
    with open(data_cih, 'r', errors='replace') as meta_file:
        with open(cleaned_cih, 'w') as meta_xml:
            #Read and clean the first line
            first_line = meta_file.readline()
            parts = first_line.split('<')
            meta_xml.write('<' + parts[-1] + '\n')
            #Go through the rest of the lines
            while True:
                line_data = meta_file.readline()
                meta_xml.write(line_data + '\n')
                if line_data.startswith('</cih>'):
                    break
    return cleaned_cih

                    
def parse_cih_xml(cleaned_cih):
    '''Parses a cleaned CIH file to find critical data for reading MRAW.
    This includes the total number of frames, the number of rows and columns,
    and the bit depth.
    Input
    cleaned_cih: Path object with the XML file holding the CIH file data
    Output
    rows: # of rows in the image
    columns: # of columns in the image
    recorded_frames: # of frames recorded
    bits: # of bits in the recorded data
    '''
    xml_data = ET.parse(cleaned_cih)
    xml_root = xml_data.getroot()
    recorded_frames = int(xml_root.find('frameInfo').find('recordedFrame').text)
    rows = int(xml_root.find('imageFileInfo').find('resolution').find('height').text)
    columns = int(xml_root.find('imageFileInfo').find('resolution').find('width').text)
    bits = int(xml_root.find('imageDataInfo').find('colorInfo').find('bit').text)
    return rows, columns, recorded_frames, bits


@nb.njit(nb.uint16[::1](nb.uint8[::1],nb.uint16[::1]),fastmath=True,parallel=True,cache=True)
def nb_read_uint12_prealloc(data_chunk,out):
    """data_chunk is a contigous 1D array of uint8 data)
    eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)
    Taken from a code sample from StackOverflow using numba: 
    https://stackoverflow.com/questions/44735756/python-reading-12-bit-binary-files
    """

    #ensure that the data_chunk has the right length
    assert np.mod(data_chunk.shape[0],3)==0
    assert out.shape[0]==data_chunk.shape[0]//3*2

    for i in nb.prange(data_chunk.shape[0]//3):
        fst_uint8=np.uint16(data_chunk[i*3])
        mid_uint8=np.uint16(data_chunk[i*3+1])
        lst_uint8=np.uint16(data_chunk[i*3+2])

        out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out


def read_mraw_frame(mraw_fname, rows, columns, frame_number):
    frame_bytes = int(rows * columns * 3 / 2)
    seek_bytes = int(frame_bytes * frame_number)
    encoded_data = np.fromfile(mraw_fname, dtype = np.uint8, count = frame_bytes, offset = seek_bytes)
    final_data = np.empty((rows * columns,), dtype = np.uint16)
    final_data = nb_read_uint12_prealloc(encoded_data, final_data)
    return final_data.reshape(rows, columns)