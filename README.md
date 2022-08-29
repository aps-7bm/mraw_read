# mraw_read
Code to parse Photron 12-bit packed monochrome MRAW files

This code is meant to parse the meta data for Photron MRAW image files
saved in a packed format, where the data for two 12-bit pixels is
saved as three 8-bit pixels. 

It also includes code to parse the Photron CIH file.  The file can
be rewritten as a valid XML file, then parsed to find the image 
size, number of images in the MRAW file, and the bit depth. 
