#!/bin/bash

split-pipe \
--mode all \
--chemistry v3 \
--genome_dir /cfs/klemming/home/m/mdalman/mercedes/Pipelines/parse_pipeline/genomes/hg38/ \
--fq1 /cfs/klemming/home/m/mdalman/mercedes/Pipelines/parse_pipeline/run_P33552/files_no_split/P33552_1001/P33552_1001_R1.fastq.gz \
--fq2 /cfs/klemming/home/m/mdalman/mercedes/Pipelines/parse_pipeline/run_P33552/files_no_split/P33552_1001/P33552_1001_R2.fastq.gz \
--output_dir /cfs/klemming/home/m/mdalman/mercedes/Pipelines/parse_pipeline/run_P33552/P33552 \
--samp_sltab /cfs/klemming/home/m/mdalman/mercedes/Pipelines/parse_pipeline/run_P33552/SampleLoadingTable.xlsm
