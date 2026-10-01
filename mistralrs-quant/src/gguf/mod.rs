    ) -> Result<String> {
        let dtype = crate::uqff::u32_scalar_with_suffix(tensors, prefix, "weight.dtype")?;
        Ok(gguf_dtype_label(dtype))
    }

    pub fn isq_type_from_uqff_dtype(dtype: u32) -> Result<IsqType> {
        IsqType::try_from(ggml_dtype_from_uqff_code(dtype)?)
    }

    pub(crate) fn block_size_from_uqff_dtype(dtype: u32) -> Result<usize> {
        Ok(ggml_dtype_from_uqff_code(dtype)?.block_size())
    }

    pub fn from_raw_uqff(
        dtype: u32,
        tensor_data: Vec<u8>,
        dims: Vec<usize>,
        b: Option<Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let dtype = ggml_dtype_from_uqff_code(dtype)?;
        let w = qtensor_from_ggml(dtype, &tensor_data, dims, device)?;
        // from_arc densifies float fallback entries, matching what ISQ produces at load
        Ok(Self::from_parts(
            QMatMul::from_arc(w.into())?,
            b,
            crate::ImatrixLayerStats::empty(),
        ))
    }

    /// Construct without `QMatMul::from_arc`: densifying would bypass the gather kernels