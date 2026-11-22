import { Box, Typography } from '@mui/material';
import { FileUpload } from '../../controls/FileUpload';
import { GenerateButton } from '../../controls/GenerateButton';
import { GridBottom, GridLeft, GridRight, Layout } from '../../controls/Layout';
import { SelectInput } from '../../controls/SelectInput';
import { TextInput } from '../../controls/TextInput';
import { ToggleInput } from '../../controls/ToggleInput';
import { UploadType } from '../../controls/UploadType';
import { VideoResult } from '../../controls/VideoResult';
import { WFTab } from '../../WFTab';

const demoPrompt =
    'A single continuous cinematic shot of soft ocean waves rolling onto a quiet beach at sunrise. Natural golden light, gentle surf and wind ambience, stable camera.';

const modes = [
    { value: 'Text to video', text: 'Text to video (FL2VA, default)' },
    { value: 'First frame to video', text: 'First image to video (FL2VA)' },
    { value: 'Last frame to video', text: 'Last image to video (FL2VA)' },
    { value: 'First and last frame', text: 'First + last image to video (FL2VA)' },
    { value: 'Timeline guides', text: 'Timed image / video guides (FL2VA)' },
    { value: 'Reference to video', text: 'Reference image / audio / video (Ref2VA)' },
];

const H3TurboRuntime = () => (
    <Layout>
        <GridLeft>
            <Typography variant='subtitle2'>H3 Turbo Runtime · native audio + video</Typography>
            <Typography variant='body2' color='text.secondary'>
                Default: 5s · 832×480 · 24 FPS · 4-step Fast · automatic backbone.
                FL2VA is used for text, first/last frames and timed guides;
                Ref2VA is selected when reference assets are provided.
            </Typography>
            <SelectInput name='mode' label='Generation task' defaultValue='Text to video'
                choices={modes} fullWidth sx={{ mt: 2 }} />
            <TextInput name='prompt' label='Scene, motion and sound'
                defaultValue={demoPrompt} multiline minRows={4} />
            <Typography variant='subtitle1' sx={{ mt: 2 }}>Generation parameters</Typography>
            <TextInput name='duration_s' type='number' label='Duration in seconds'
                defaultValue={5} />
            <TextInput name='steps' type='number' label='Sampling steps'
                defaultValue={4} />
            <SelectInput name='strategy' label='GPU execution strategy'
                defaultValue='fast' choices={[{ value: 'fast', text: 'Fast' },
                                              { value: 'stream', text: 'Stream' }]} />
            <TextInput name='seed' type='number' label='Seed' defaultValue={4106} />
            <SelectInput name='canvas_mode' label='Canvas size method'
                defaultValue='explicit' choices={[
                    { value: 'explicit', text: 'Explicit width × height (default)' },
                    { value: 'short_edge_ratio', text: 'Short edge + aspect ratio (official style)' },
                ]} />
            <TextInput name='width' type='number' label='Width (pixels)'
                defaultValue={832} />
            <TextInput name='height' type='number' label='Height (pixels)'
                defaultValue={480} />
            <TextInput name='short_edge' type='number' label='Short edge (official default 768)'
                defaultValue={768} />
            <TextInput name='aspect_ratio' label='Aspect ratio (e.g. 16:9, 9:16, auto)'
                defaultValue='16:9' />
            <TextInput name='length_frames' type='number' label='Native frame count override (0 = use seconds)'
                defaultValue={0} />
            <TextInput name='output_fps' type='number' label='Output FPS (native generation uses 24)'
                defaultValue={24} />
            <SelectInput name='variant' label='Model override (advanced)'
                defaultValue='auto' choices={[
                    { value: 'auto', text: 'Auto: choose from media conditions' },
                    { value: 'fl2va', text: 'Force FL2VA' },
                    { value: 'ref2va', text: 'Force Ref2VA' },
                ]} />
            <Typography variant='subtitle1' sx={{ mt: 2 }}>Optional frame guides</Typography>
            <Typography variant='body2' color='text.secondary'>
                Leave empty for text-only generation. Upload first and/or last frame;
                the middle frame may be placed at any time within the duration.
            </Typography>
            <FileUpload name='first_frame' type={UploadType.IMAGE}
                backgroundBackup directInput />
            <FileUpload name='last_frame' type={UploadType.IMAGE}
                backgroundBackup directInput />
            <FileUpload name='middle_frame' type={UploadType.IMAGE}
                backgroundBackup directInput />
            <TextInput name='middle_at_seconds' type='number'
                label='Middle image at (seconds)' defaultValue={2} />
            <Typography variant='subtitle1' sx={{ mt: 2 }}>Optional character / voice references</Typography>
            <Typography variant='body2' color='text.secondary'>
                Add a reference image and optional voice clip as in the Nailong Ref2VA example;
                these are identity or timbre guides, not mandatory start frames.
            </Typography>
            <FileUpload name='reference_image' type={UploadType.IMAGE}
                backgroundBackup directInput />
            <FileUpload name='reference_audio' type={UploadType.AUDIO}
                backgroundBackup directInput />
            <FileUpload name='reference_video' type={UploadType.VIDEO}
                backgroundBackup directInput />
            <ToggleInput name='reference_video_audio'
                label='Use audio from reference video' defaultValue={false} />
            <SelectInput name='ref_image_size' label='Reference image preprocessing'
                defaultValue='match' choices={[
                    { value: 'match', text: 'Match target pixel area (default)' },
                    { value: 'max', text: 'Use original H3 large reference budget' },
                ]} />
            <Typography variant='subtitle1' sx={{ mt: 2 }}>Runtime advanced tuning</Typography>
            <TextInput name='head_group' type='number' label='Attention head group' defaultValue={7} />
            <TextInput name='reserve_vram' type='number' label='Reserved VRAM GiB (0 = runtime default)' defaultValue={0} />
            <TextInput name='adapter_file' label='Optional LoRA adapter path' defaultValue='' />
            <TextInput name='recipe_file' label='Optional sampler recipe path' defaultValue='' />
            <Box sx={{ mt: 2 }}>
                <Typography variant='subtitle1'>Advanced multi-guide input (optional)</Typography>
                <TextInput name='guides_json' label='Other timed guides JSON'
                    multiline minRows={2} defaultValue='[]' />
                <TextInput name='references_json' label='Additional references JSON'
                    multiline minRows={2} defaultValue='{}' />
            </Box>
        </GridLeft>
        <GridRight><VideoResult fps={24} /></GridRight>
        <GridBottom>
            <GenerateButton serverManagedInputs requiredControls={['prompt']} />
        </GridBottom>
    </Layout>
);

export const H3TurboRuntimeTab = (
    <WFTab label='H3 Turbo Runtime' value='H3 Turbo Runtime'
        group='H3' content={<H3TurboRuntime />} />
);
