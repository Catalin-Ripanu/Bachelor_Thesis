#!/bin/bash

scp -r $1 catalin.ripanu@fep8.grid.pub.ro:./Bachelor_Thesis/$2
scp catalin.ripanu@fep8.grid.pub.ro:/export/home/acs/stud/c/catalin.ripanu/Bachelor_Thesis/quantum_models/*.pdf ./new_pdf_graphs
scp catalin.ripanu@fep8.grid.pub.ro:/export/home/acs/stud/c/catalin.ripanu/Bachelor_Thesis/quantum_models/generated_images/*.jpg ./qrkt_gan
