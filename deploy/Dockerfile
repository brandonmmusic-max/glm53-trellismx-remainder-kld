# TrellisMX r27 + RP2 for GLM-5.3-Flash.
# RP2: row-packed two-term (hi + lo) E4M3/UE8M0-K32 activation carrier at both P8 MoE hops on the direct decode
# kernels (<= 16 tokens per step). EPI-PAR: bit-identical parallel epilogue in the M1 FC1 kernel. Host-side guards
# reject untested flag combinations. Study, patches and measurements:
# https://github.com/brandonmmusic-max/glm53-trellismx-remainder-kld
FROM verdictai/trellismx@sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf
RUN rm -rf /opt/glm53-flash/b12x
COPY b12x /opt/glm53-flash/b12x
COPY serve-rp2.sh /release/serve-rp2.sh
ENV B12X_P8_DOWN_REMAINDER=rp2 \
    B12X_P8_EPI_PAR=1
LABEL org.opencontainers.image.title="TrellisMX GLM-5.3-Flash r27 + RP2" \
      org.opencontainers.image.description="TrellisMX r27 P8 MoE kernels with the RP2 two-term FP8 activation carrier (decode kernels) and the bit-identical EPI-PAR M1 epilogue; FP8 MLA KV by default in serve-rp2.sh" \
      org.opencontainers.image.source="https://github.com/brandonmmusic-max/glm53-trellismx-remainder-kld" \
      org.opencontainers.image.version="glm53-flash-p8-r27-rp2-20260926" \
      org.opencontainers.image.base.name="verdictai/trellismx:glm53-flash-p8-r27-reference-20260909" \
      org.opencontainers.image.base.digest="sha256:ca6b80188dce154b91f49108b7d87792d2ba6328935afc71b44d1c0e6f6a1adf"
ENTRYPOINT ["/bin/bash", "/release/serve-rp2.sh"]
